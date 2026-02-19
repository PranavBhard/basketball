"""
ESPN Sync Service - Orchestrates fetching data from ESPN API and saving to MongoDB.

Composes ESPNClient (fetch) + repository layer (save) to provide:
- fetch_and_save_games: Pull game data for a date range and upsert to DB
- refresh_venues: Refresh venue location data from JSON
- refresh_players: Refresh player metadata from player_stats collection

This is the core-layer replacement for subprocess calls to the legacy ESPN CLI script.

Usage:
    from bball.services.espn_sync import fetch_and_save_games, refresh_venues, refresh_players
    from bball.mongo import Mongo
    from bball.league_config import load_league_config

    league = load_league_config('nba')
    db = Mongo().db

    stats = fetch_and_save_games(db, league, start_date, end_date)
    refresh_venues(db, league)
    refresh_players(db, league)
"""

import json
import os
import re
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from pytz import timezone, utc

from bball.data.espn_client import ESPNClient

if TYPE_CHECKING:
    from bball.league_config import LeagueConfig


# Threading constants for batch processing
THREADING_THRESHOLD = 2500
CHUNK_SIZE = 500
MAX_WORKERS = 10


# ---------------------------------------------------------------------------
# Helper: parse ESPN UTC timestamps
# ---------------------------------------------------------------------------

def _parse_espn_utc_to_eastern(date_str: str, league: "LeagueConfig" = None) -> Optional[datetime]:
    """Parse ESPN UTC date string and convert to league timezone (default Eastern)."""
    if not date_str:
        return None
    try:
        dt_utc = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        tz_name = league.timezone if league else 'America/New_York'
        return dt_utc.astimezone(timezone(tz_name))
    except (ValueError, TypeError):
        return None


def _parse_gametime(event_date_str: str) -> Optional[datetime]:
    """Parse event date string (already in UTC) and return as UTC datetime."""
    if not event_date_str:
        return None
    try:
        try:
            iso_str = event_date_str.replace('Z', '+00:00')
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = utc.localize(dt)
            elif dt.tzinfo != utc:
                dt = dt.astimezone(utc)
            return dt
        except (ValueError, AttributeError):
            pass

        if 'T' in event_date_str:
            date_part = event_date_str.split('T')[0]
            time_part = event_date_str.split('T')[1]
            if time_part.endswith('Z'):
                time_part = time_part[:-1]
            elif '+' in time_part:
                time_part = time_part.split('+')[0]
            elif '-' in time_part and time_part.count('-') > 1:
                parts = time_part.rsplit('-', 1)
                if len(parts) == 2 and ':' in parts[1]:
                    time_part = parts[0]

            dt_str = f"{date_part}T{time_part}"
            try:
                dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                try:
                    dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%f')
                except ValueError:
                    dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M')
            return utc.localize(dt)
        else:
            dt = datetime.strptime(event_date_str, '%Y-%m-%d')
            return utc.localize(dt)
    except Exception as e:
        print(f"  Warning: Could not parse gametime '{event_date_str}': {e}")
        return None


# ---------------------------------------------------------------------------
# Extraction helpers (from ESPN JSON responses)
# ---------------------------------------------------------------------------

def _extract_team_stats(team_data: Dict, boxscore: Dict) -> Dict:
    """Extract team stats from boxscore data."""
    stats = {}
    team_id = team_data.get('id')
    team_stats = None
    for team in boxscore.get('teams', []):
        if team.get('team', {}).get('id') == team_id:
            team_stats = team.get('statistics', [])
            break
    if not team_stats:
        return stats

    stat_map = {
        'fieldGoalsMade-fieldGoalsAttempted': ('FG_made', 'FG_att'),
        'fieldGoalPct': 'FGp',
        'threePointFieldGoalsMade-threePointFieldGoalsAttempted': ('three_made', 'three_att'),
        'threePointFieldGoalPct': 'three_percent',
        'freeThrowsMade-freeThrowsAttempted': ('FT_made', 'FT_att'),
        'freeThrowPct': 'FTp',
        'totalRebounds': 'total_reb',
        'offensiveRebounds': 'off_reb',
        'defensiveRebounds': 'def_reb',
        'assists': 'assists',
        'steals': 'steals',
        'blocks': 'blocks',
        'turnovers': 'TO',
        'turnoverPoints': 'pts_off_TO',
        'fastBreakPoints': 'fast_break_pts',
        'pointsInPaint': 'pts_in_paint',
        'fouls': 'PF',
    }

    for stat in team_stats:
        name = stat.get('name')
        display_value = stat.get('displayValue', '')
        if name in stat_map:
            mapping = stat_map[name]
            if isinstance(mapping, tuple):
                if '-' in display_value:
                    parts = display_value.split('-')
                    if len(parts) == 2:
                        stats[mapping[0]] = int(parts[0])
                        stats[mapping[1]] = int(parts[1])
            else:
                if name in ('fieldGoalPct', 'threePointFieldGoalPct', 'freeThrowPct'):
                    pct_str = display_value.replace('%', '').strip()
                    if pct_str:
                        stats[mapping] = float(pct_str) / 100.0
                else:
                    if display_value.isdigit():
                        stats[mapping] = int(display_value)

    if 'FG_made' in stats and 'FG_att' in stats and stats['FG_att'] > 0:
        stats['shooting_metric'] = (stats['FG_made'] + (0.5 * stats.get('three_made', 0))) / float(stats['FG_att'])

    return stats


def _extract_season_series(game_summary: Dict) -> Optional[Dict]:
    """Extract season series information."""
    seasonseries = game_summary.get('seasonseries', [])
    if not seasonseries:
        return None
    series = seasonseries[0]
    return {
        'type': series.get('type'),
        'title': series.get('title'),
        'description': series.get('description'),
        'summary': series.get('summary'),
        'series_label': series.get('seriesLabel')
    }


def _extract_venue_info(game_summary: Dict) -> Optional[Dict]:
    """Extract venue information from game summary."""
    game_info = game_summary.get('gameInfo', {})
    venue = game_info.get('venue')
    if not venue:
        return None
    venue_guid = venue.get('guid')
    if not venue_guid:
        return None
    return {
        'venue_guid': venue_guid,
        'id': venue.get('id'),
        'fullName': venue.get('fullName'),
        'shortName': venue.get('shortName'),
        'address': venue.get('address', {}),
        'grass': venue.get('grass', False),
        'images': venue.get('images', [])
    }


def _extract_odds(game_summary: Dict) -> Optional[Dict]:
    """Extract odds from game summary pickcenter."""
    pickcenter = game_summary.get('pickcenter', [])
    if not pickcenter:
        return None
    odds = pickcenter[0]
    pregame_lines = {}
    if odds.get('overUnder') is not None:
        pregame_lines['over_under'] = odds['overUnder']
    if odds.get('spread') is not None:
        pregame_lines['spread'] = odds['spread']
    home_odds = odds.get('homeTeamOdds', {})
    if home_odds and home_odds.get('moneyLine') is not None:
        pregame_lines['home_ml'] = home_odds['moneyLine']
    away_odds = odds.get('awayTeamOdds', {})
    if away_odds and away_odds.get('moneyLine') is not None:
        pregame_lines['away_ml'] = away_odds['moneyLine']
    return pregame_lines if pregame_lines else None


def _extract_player_stats(
    player_data: Dict,
    game_id: str,
    game_date: str,
    team_name: str,
    home: bool,
    opponent: str,
    stat_group: Dict,
    season: str = None,
    team_id: str = None,
    opponent_id: str = None,
    league: "LeagueConfig" = None
) -> Optional[Dict]:
    """Extract player stats from API response."""
    athlete = player_data.get('athlete', {})
    stats_array = player_data.get('stats', [])

    player_id = athlete.get('id')
    if not player_id:
        return None

    stat_keys = stat_group.get('keys', [])
    stat_values = stats_array

    if not season:
        game_date_obj = datetime.strptime(game_date, '%Y-%m-%d').date()
        cutover_month = (league.season_rules.get('season_cutover_month', 8) if league else 8)
        if game_date_obj.month > cutover_month:
            season = f'{game_date_obj.year}-{game_date_obj.year + 1}'
        else:
            season = f'{game_date_obj.year - 1}-{game_date_obj.year}'

    player_stats = {
        'player_id': str(player_id),
        'game_id': game_id,
        'date': game_date,
        'season': season,
        'team': team_name,
        'home': home,
        'opponent': opponent,
        'starter': player_data.get('starter', False),
        'active': player_data.get('active', True),
        'didNotPlay': player_data.get('didNotPlay', False),
        'guid': athlete.get('guid'),
        'short_name': athlete.get('shortName'),
        'player_name': athlete.get('displayName'),
    }

    headshot_url = athlete.get('headshot', {}).get('href') if athlete.get('headshot') else None
    if headshot_url:
        player_stats['_headshot_for_players'] = headshot_url

    pos_name = athlete.get('position', {}).get('name') if athlete.get('position') else None
    pos_display_name = athlete.get('position', {}).get('displayName') if athlete.get('position') else None
    if pos_name or pos_display_name:
        player_stats['_position_for_players'] = {
            'pos_name': pos_name,
            'pos_display_name': pos_display_name
        }

    if team_id is not None:
        player_stats['team_id'] = str(team_id)
    if opponent_id is not None:
        player_stats['opponent_id'] = str(opponent_id)

    stats_dict = {}
    if stat_keys and stat_values:
        for key, value in zip(stat_keys, stat_values):
            if key == 'minutes':
                if ':' in str(value):
                    parts = str(value).split(':')
                    stats_dict['min'] = int(parts[0]) + int(parts[1]) / 60.0
                else:
                    try:
                        stats_dict['min'] = float(value)
                    except Exception:
                        stats_dict['min'] = 0
            elif key == 'points':
                stats_dict['pts'] = int(value) if str(value).isdigit() else 0
            elif key == 'fieldGoalsMade-fieldGoalsAttempted':
                if '-' in str(value):
                    parts = str(value).split('-')
                    stats_dict['fg_made'] = int(parts[0]) if parts[0].isdigit() else 0
                    stats_dict['fg_att'] = int(parts[1]) if parts[1].isdigit() else 0
            elif key == 'threePointFieldGoalsMade-threePointFieldGoalsAttempted':
                if '-' in str(value):
                    parts = str(value).split('-')
                    stats_dict['three_made'] = int(parts[0]) if parts[0].isdigit() else 0
                    stats_dict['three_att'] = int(parts[1]) if parts[1].isdigit() else 0
            elif key == 'freeThrowsMade-freeThrowsAttempted':
                if '-' in str(value):
                    parts = str(value).split('-')
                    stats_dict['ft_made'] = int(parts[0]) if parts[0].isdigit() else 0
                    stats_dict['ft_att'] = int(parts[1]) if parts[1].isdigit() else 0
            elif key == 'rebounds':
                stats_dict['reb'] = int(value) if str(value).isdigit() else 0
            elif key == 'offensiveRebounds':
                stats_dict['oreb'] = int(value) if str(value).isdigit() else 0
            elif key == 'defensiveRebounds':
                stats_dict['dreb'] = int(value) if str(value).isdigit() else 0
            elif key == 'assists':
                stats_dict['ast'] = int(value) if str(value).isdigit() else 0
            elif key == 'turnovers':
                stats_dict['to'] = int(value) if str(value).isdigit() else 0
            elif key == 'steals':
                stats_dict['stl'] = int(value) if str(value).isdigit() else 0
            elif key == 'blocks':
                stats_dict['blk'] = int(value) if str(value).isdigit() else 0
            elif key == 'fouls':
                stats_dict['pf'] = int(value) if str(value).isdigit() else 0
            elif key == 'plusMinus':
                pm_str = str(value).replace('+', '').replace('-', '')
                if pm_str.lstrip('-').isdigit():
                    stats_dict['plus_minus'] = int(pm_str) if not str(value).startswith('-') else -int(pm_str)

    player_stats['stats'] = stats_dict
    return player_stats


# ---------------------------------------------------------------------------
# process_game — process a single ESPN game and upsert to MongoDB
# ---------------------------------------------------------------------------

def _process_game(
    game_id: str,
    game_date: date,
    db,
    league: "LeagueConfig",
    espn_client: ESPNClient,
    team_only: bool = False,
    players_only: bool = False,
    dry_run: bool = False,
    event: Optional[Dict] = None,
    quiet: bool = False
) -> Tuple[bool, int]:
    """Process a single game: fetch data and store in MongoDB."""
    from bball.data.league_db_proxy import LeagueDbProxy
    league_db = LeagueDbProxy(db, league)

    game_summary = espn_client.get_game_summary(game_id)
    if not game_summary:
        return False, 0

    header = game_summary.get('header', {})
    competitions0 = header.get('competitions', [{}])[0]
    competitors = competitions0.get('competitors', [])

    status_type = (competitions0.get('status') or {}).get('type') or {}
    is_completed = bool(status_type.get('completed', False))

    if len(competitors) != 2:
        print(f"  Warning: Game {game_id} has {len(competitors)} competitors, skipping")
        return False, 0

    home_team = None
    away_team = None
    home_score = 0
    away_score = 0
    home_linescores = []
    away_linescores = []

    for comp in competitors:
        if comp.get('homeAway') == 'home':
            home_team = comp.get('team', {})
            home_score = int(comp.get('score', 0) or 0)
            home_linescores = comp.get('linescores', [])
        else:
            away_team = comp.get('team', {})
            away_score = int(comp.get('score', 0) or 0)
            away_linescores = comp.get('linescores', [])

    if not home_team or not away_team:
        print(f"  Warning: Could not determine home/away for game {game_id}")
        return False, 0

    home_name = home_team.get('abbreviation', '').upper()
    away_name = away_team.get('abbreviation', '').upper()

    if not home_name or not away_name:
        print(f"  Warning: Missing team abbreviations for game {game_id}")
        return False, 0

    if not is_completed:
        detail = status_type.get('shortDetail') or status_type.get('detail') or status_type.get('description') or 'Not completed'
        print(f"  Skipping {away_name}@{home_name} ({game_id}): game not completed yet ({detail})")
        return False, 0

    date_str = game_date.strftime('%Y-%m-%d')
    year, month, day = game_date.year, game_date.month, game_date.day

    # Extract ESPN-provided season year
    def _extract_espn_season_year() -> Optional[int]:
        comp_season = competitions0.get('season')
        if isinstance(comp_season, int):
            return comp_season
        if isinstance(comp_season, str) and comp_season.isdigit():
            return int(comp_season)
        header_season = header.get('season')
        if isinstance(header_season, dict):
            season_year = header_season.get('year')
            if isinstance(season_year, int):
                return season_year
            if isinstance(season_year, str) and season_year.isdigit():
                return int(season_year)
        if isinstance(comp_season, dict):
            season_year = comp_season.get('year')
            if isinstance(season_year, int):
                return season_year
            if isinstance(season_year, str) and season_year.isdigit():
                return int(season_year)
        return None

    espn_season_year = _extract_espn_season_year()

    if espn_season_year:
        season = f'{espn_season_year - 1}-{espn_season_year}'
    else:
        cutover_month = (league.season_rules.get('season_cutover_month', 8) if league else 8)
        if month > cutover_month:
            season = f'{year}-{year + 1}'
        else:
            season = f'{year - 1}-{year}'

    # Determine game type
    def _extract_season_type_info() -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
        season_obj = competitions0.get('season') or header.get('season') or {}
        if not isinstance(season_obj, dict):
            season_obj = {}
        slug = season_obj.get('slug')
        t = season_obj.get('type')
        type_id = None
        type_name = None
        type_abbrev = None
        if isinstance(t, dict):
            raw_type = t.get('type') if t.get('type') is not None else t.get('id')
            try:
                type_id = int(raw_type) if raw_type is not None else None
            except Exception:
                type_id = None
            type_name = t.get('name')
            type_abbrev = t.get('abbreviation')
        else:
            try:
                type_id = int(t) if t is not None else None
            except Exception:
                type_id = None
        if not slug:
            s2 = header.get('season') if isinstance(header.get('season'), dict) else {}
            slug = (s2 or {}).get('slug') if isinstance(s2, dict) else None
        return type_id, (str(slug) if slug is not None else None), (str(type_name) if type_name is not None else None), (str(type_abbrev) if type_abbrev is not None else None)

    def _extract_tournament_info() -> Tuple[Optional[bool], Optional[Dict]]:
        candidates = [
            competitions0.get('tournament'),
            header.get('tournament'),
            game_summary.get('tournament'),
        ]
        for c in candidates:
            if isinstance(c, dict):
                is_t = c.get('isTournament')
                if isinstance(is_t, bool):
                    return is_t, c
            if isinstance(c, bool):
                return c, None
        for c in [competitions0.get('isTournament'), header.get('isTournament')]:
            if isinstance(c, bool):
                return c, None
        return None, None

    season_type_id, season_type_slug, season_type_name, season_type_abbrev = _extract_season_type_info()
    is_tournament, tournament_obj = _extract_tournament_info()

    season_cfg = (league.season_rules if league else {}) or {}
    game_type_map = season_cfg.get('game_type_map') or {}
    game_type = None
    if season_type_id is not None:
        game_type = game_type_map.get(season_type_id) or game_type_map.get(str(season_type_id))

    if not game_type:
        if season_type_id == 1:
            game_type = 'preseason'
        elif season_type_id == 3:
            game_type = 'playoffs' if (league and league.league_id == 'nba') else 'postseason'
        else:
            game_type = 'regseason'

    if bool(season_cfg.get('playin_text_override', False)):
        text_fields = [
            str(header.get('name') or ''),
            str(header.get('shortName') or ''),
            str(competitions0.get('type') or ''),
            str((competitions0.get('status') or {}).get('type', {}).get('detail') or ''),
        ]
        text_blob = ' '.join([t for t in text_fields if t]).lower()
        if 'play-in' in text_blob:
            game_type = 'playin'

    if is_tournament is True and season_type_id == 3:
        tournament_game_type = season_cfg.get('postseason_tournament_game_type')
        if isinstance(tournament_game_type, str) and tournament_game_type.strip():
            game_type = tournament_game_type.strip()

    venue_data = _extract_venue_info(game_summary)
    venue_guid = None
    if venue_data:
        venue_guid = venue_data['venue_guid']
        if not dry_run:
            league_db.nba_venues.update_one(
                {'venue_guid': venue_guid},
                {'$set': venue_data},
                upsert=True
            )

    series_data = _extract_season_series(game_summary)
    pregame_lines = _extract_odds(game_summary)

    gametime = None
    seasonseries = game_summary.get('seasonseries', [])
    if seasonseries and len(seasonseries) > 0:
        first_series = seasonseries[0]
        events = first_series.get('events', [])
        for evt in events:
            evt_id = evt.get('id')
            if evt_id and (str(evt_id) == str(game_id) or evt_id == game_id):
                event_date_str = evt.get('date')
                if event_date_str:
                    gametime = _parse_gametime(event_date_str)
                    break

    if not gametime:
        if event:
            event_date_str = event.get('date')
            if event_date_str:
                gametime = _parse_gametime(event_date_str)
        else:
            hdr = game_summary.get('header', {})
            competitions = hdr.get('competitions', [])
            if competitions:
                event_date_str = competitions[0].get('date')
                if event_date_str:
                    gametime = _parse_gametime(event_date_str)

    game_data = {
        'game_id': game_id,
        'date': date_str,
        'year': year,
        'month': month,
        'day': day,
        'season': season,
        'espn_season_year': espn_season_year,
        'game_type': game_type,
        'season_type_id': season_type_id,
        'season_type_slug': season_type_slug,
        'season_type_name': season_type_name,
        'season_type_abbrev': season_type_abbrev,
        'isTournament': is_tournament,
        'neutralSite': bool(competitions0.get('neutralSite', False)),
        'description': (header.get('name') or header.get('shortName') or 'Regular Season'),
        'homeWon': home_score > away_score,
        'OT': False,
        'venue_guid': venue_guid,
        'espn_link': f"https://www.espn.com/{league.league_id}/boxscore/_/gameId/{game_id}",
    }

    if gametime:
        game_data['gametime'] = gametime
    if series_data:
        game_data.update(series_data)
    if pregame_lines:
        game_data['pregame_lines'] = pregame_lines

    boxscore = game_summary.get('boxscore', {})
    teams = boxscore.get('teams', [])

    home_stats = {}
    away_stats = {}

    for team in teams:
        team_info = team.get('team', {})
        t_id = team_info.get('id')
        if t_id == home_team.get('id'):
            home_stats = _extract_team_stats(team_info, {'teams': [team]})
            home_stats['team_id'] = str(t_id) if t_id is not None else None
            home_stats['name'] = home_name
            home_stats['points'] = home_score
        elif t_id == away_team.get('id'):
            away_stats = _extract_team_stats(team_info, {'teams': [team]})
            away_stats['team_id'] = str(t_id) if t_id is not None else None
            away_stats['name'] = away_name
            away_stats['points'] = away_score

    if 'FG_att' in home_stats and 'FG_att' in away_stats:
        possessions_home = home_stats['FG_att'] - home_stats.get('off_reb', 0) + home_stats.get('TO', 0) + (0.4 * home_stats.get('FT_att', 0))
        possessions_away = away_stats['FG_att'] - away_stats.get('off_reb', 0) + away_stats.get('TO', 0) + (0.4 * away_stats.get('FT_att', 0))
        possessions = (possessions_home + possessions_away) / 2.0

        if possessions > 0:
            home_stats['TO_metric'] = home_stats.get('TO', 0) / float(possessions)
            away_stats['TO_metric'] = away_stats.get('TO', 0) / float(possessions)

        if 'off_reb' in home_stats and 'def_reb' in away_stats:
            total_reb = home_stats.get('off_reb', 0) + away_stats.get('def_reb', 0)
            if total_reb > 0:
                home_stats['off_reb_metric'] = home_stats.get('off_reb', 0) / float(total_reb)

        if 'off_reb' in away_stats and 'def_reb' in home_stats:
            total_reb = away_stats.get('off_reb', 0) + home_stats.get('def_reb', 0)
            if total_reb > 0:
                away_stats['off_reb_metric'] = away_stats.get('off_reb', 0) / float(total_reb)

    if home_linescores and away_linescores:
        home_stats['points1q'] = int(home_linescores[0].get('displayValue', 0)) if len(home_linescores) > 0 else 0
        home_stats['points2q'] = int(home_linescores[1].get('displayValue', 0)) if len(home_linescores) > 1 else 0
        home_stats['points3q'] = int(home_linescores[2].get('displayValue', 0)) if len(home_linescores) > 2 else 0
        home_stats['points4q'] = int(home_linescores[3].get('displayValue', 0)) if len(home_linescores) > 3 else 0

        away_stats['points1q'] = int(away_linescores[0].get('displayValue', 0)) if len(away_linescores) > 0 else 0
        away_stats['points2q'] = int(away_linescores[1].get('displayValue', 0)) if len(away_linescores) > 1 else 0
        away_stats['points3q'] = int(away_linescores[2].get('displayValue', 0)) if len(away_linescores) > 2 else 0
        away_stats['points4q'] = int(away_linescores[3].get('displayValue', 0)) if len(away_linescores) > 3 else 0

        ot_periods = len(home_linescores) - 4
        if ot_periods > 0:
            game_data['OT'] = True
            home_ot = sum(int(home_linescores[i].get('displayValue', 0)) for i in range(4, len(home_linescores)))
            away_ot = sum(int(away_linescores[i].get('displayValue', 0)) for i in range(4, len(away_linescores)))
            home_stats['pointsOT'] = home_ot
            away_stats['pointsOT'] = away_ot
        else:
            game_data['OT'] = False
            home_stats['pointsOT'] = 0
            away_stats['pointsOT'] = 0

    game_data['homeTeam'] = home_stats
    game_data['awayTeam'] = away_stats

    if not players_only and not dry_run:
        query = {'game_id': game_id}
        flat_update = {}
        for key, value in game_data.items():
            if key in ('homeTeam', 'awayTeam') and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    flat_update[f'{key}.{nested_key}'] = nested_value
            else:
                flat_update[key] = value
        league_db.stats_nba.update_one(query, {'$set': flat_update}, upsert=True)

    if not quiet:
        if dry_run:
            print(f"  [DRY RUN] Would store game: {away_name} @ {home_name} ({away_score}-{home_score})")
        else:
            print(f"  Stored game: {away_name} @ {home_name} ({away_score}-{home_score})")

    player_count = 0
    if not team_only:
        players = boxscore.get('players', [])
        home_team_id = home_team.get('id')
        away_team_id = away_team.get('id')
        include_team_ids = league.include_team_id if league else False

        for team_player_data in players:
            team_info = team_player_data.get('team', {})
            t_id = team_info.get('id')
            is_home = t_id == home_team_id
            team_name = home_name if is_home else away_name
            opponent = away_name if is_home else home_name
            opponent_team_id = away_team_id if is_home else home_team_id

            statistics = team_player_data.get('statistics', [])
            if not statistics:
                continue

            for stat_group in statistics:
                athletes = stat_group.get('athletes', [])
                for athlete_data in athletes:
                    pstats = _extract_player_stats(
                        athlete_data, game_id, date_str, team_name, is_home, opponent, stat_group,
                        season=season,
                        team_id=t_id if include_team_ids else None,
                        opponent_id=opponent_team_id if include_team_ids else None,
                        league=league
                    )
                    if pstats:
                        if not dry_run:
                            headshot_url = pstats.pop('_headshot_for_players', None)
                            position_data = pstats.pop('_position_for_players', None)

                            pquery = {'game_id': game_id, 'player_id': pstats['player_id']}
                            league_db.stats_nba_players.update_one(pquery, {'$set': pstats}, upsert=True)

                            players_update = {
                                'player_id': pstats['player_id'],
                                'player_name': pstats.get('player_name'),
                            }
                            if headshot_url:
                                players_update['headshot'] = headshot_url
                            if position_data:
                                if position_data.get('pos_name'):
                                    players_update['pos_name'] = position_data['pos_name']
                                if position_data.get('pos_display_name'):
                                    players_update['pos_display_name'] = position_data['pos_display_name']

                            players_coll = league.collections.get('players', 'nba_players') if league else 'nba_players'
                            db[players_coll].update_one(
                                {'player_id': pstats['player_id']},
                                {'$set': players_update},
                                upsert=True
                            )
                        player_count += 1

    if not team_only and not quiet:
        if dry_run:
            print(f"    [DRY RUN] Would store {player_count} player stats")
        else:
            print(f"    Stored {player_count} player stats")

    return True, player_count


# ---------------------------------------------------------------------------
# Job tracking helpers (for large date-range syncs)
# ---------------------------------------------------------------------------

def _create_job(db, league: "LeagueConfig", total_days: int, team_only: bool, players_only: bool) -> Optional[str]:
    """Create a job tracking document for ESPN date-range sync."""
    from bball.data.league_db_proxy import LeagueDbProxy
    league_db = LeagueDbProxy(db, league)
    job_doc = {
        'type': 'espn_dates_sync',
        'progress': 0,
        'status': 'running',
        'error': None,
        'message': f'Starting ESPN date-range sync for {total_days} day(s)...',
        'metadata': {
            'total_days': total_days,
            'team_only': team_only,
            'players_only': players_only,
            'days_processed': 0,
            'games_processed': 0,
            'players_processed': 0,
            'games_skipped_date_mismatch': 0,
            'failed_games': 0,
            'failed_game_ids': [],
            'failed_game_errors': []
        },
        'created_at': datetime.now(),
        'updated_at': datetime.now()
    }
    result = league_db.jobs_nba.insert_one(job_doc)
    return str(result.inserted_id)


def _update_job(db, league: "LeagueConfig", job_id: str, processed_days: int, total_days: int, games_done: int, players_done: int, skipped: int):
    """Update job progress."""
    from bson import ObjectId
    from bball.data.league_db_proxy import LeagueDbProxy
    league_db = LeagueDbProxy(db, league)
    progress = int(100 * processed_days / total_days) if total_days > 0 else 0
    league_db.jobs_nba.update_one(
        {'_id': ObjectId(job_id)},
        {'$set': {
            'progress': progress,
            'message': f'Processed {processed_days}/{total_days} days ({progress}%)',
            'metadata.days_processed': processed_days,
            'metadata.games_processed': games_done,
            'metadata.players_processed': players_done,
            'metadata.games_skipped_date_mismatch': skipped,
            'updated_at': datetime.now()
        }}
    )


def _complete_job(db, league: "LeagueConfig", job_id: str, days_done: int, total_days: int, games_done: int, players_done: int, skipped: int):
    """Mark job as completed."""
    from bson import ObjectId
    from bball.data.league_db_proxy import LeagueDbProxy
    league_db = LeagueDbProxy(db, league)
    failed_games = 0
    try:
        doc = league_db.jobs_nba.find_one({'_id': ObjectId(job_id)}, {'metadata.failed_games': 1})
        failed_games = int(((doc or {}).get('metadata') or {}).get('failed_games') or 0)
    except Exception:
        pass
    league_db.jobs_nba.update_one(
        {'_id': ObjectId(job_id)},
        {'$set': {
            'status': 'completed',
            'progress': 100,
            'message': f'Completed: {days_done}/{total_days} days, {games_done} games ({failed_games} failures)',
            'metadata.days_processed': days_done,
            'metadata.games_processed': games_done,
            'metadata.players_processed': players_done,
            'metadata.games_skipped_date_mismatch': skipped,
            'updated_at': datetime.now()
        }}
    )


def _fail_job(db, league: "LeagueConfig", job_id: str, error: str):
    """Mark job as failed."""
    from bson import ObjectId
    from bball.data.league_db_proxy import LeagueDbProxy
    league_db = LeagueDbProxy(db, league)
    league_db.jobs_nba.update_one(
        {'_id': ObjectId(job_id)},
        {'$set': {
            'status': 'failed',
            'error': error,
            'message': f'Failed: {error}',
            'updated_at': datetime.now()
        }}
    )


def _record_failure(db, league: "LeagueConfig", job_id: str, game_id: str, error: str):
    """Record a per-game failure."""
    from bson import ObjectId
    from bball.data.league_db_proxy import LeagueDbProxy
    league_db = LeagueDbProxy(db, league)
    MAX_FAILURES = 200
    try:
        league_db.jobs_nba.update_one(
            {'_id': ObjectId(job_id)},
            {
                '$inc': {'metadata.failed_games': 1},
                '$push': {
                    'metadata.failed_game_ids': {'$each': [str(game_id)], '$slice': -MAX_FAILURES},
                    'metadata.failed_game_errors': {
                        '$each': [{'game_id': str(game_id), 'error': str(error), 'at': datetime.now().isoformat()}],
                        '$slice': -MAX_FAILURES
                    }
                },
                '$set': {'updated_at': datetime.now()}
            }
        )
    except Exception:
        pass


# ===========================================================================
# Public API
# ===========================================================================

def get_matchups_with_venue(
    year: int,
    month: int,
    day: int,
    league: "LeagueConfig" = None
) -> List[Dict]:
    """
    Fetch matchups for a date with venue information.

    Uses ESPN API to get game details including venue_guid for each matchup.

    Args:
        year, month, day: Date to fetch matchups for
        league: Optional league config (defaults to NBA)

    Returns:
        List of dicts with keys: home_team, away_team, game_id, venue_guid
    """
    from datetime import date as _date
    date_obj = _date(year, month, day)

    espn_client = ESPNClient(league=league)
    scoreboard = espn_client.get_scoreboard_site(date_obj)

    if not scoreboard:
        return []

    matchups = []
    events = scoreboard.get('events', [])

    for event in events:
        game_id = event.get('id')
        if not game_id:
            continue

        competitions = event.get('competitions', [])
        if not competitions:
            continue
        competitors = competitions[0].get('competitors', [])

        home_team = None
        away_team = None
        for comp in competitors:
            team_info = comp.get('team', {})
            abbrev = team_info.get('abbreviation', comp.get('abbreviation', '')).upper()
            if comp.get('homeAway') == 'home':
                home_team = abbrev
            else:
                away_team = abbrev

        if not home_team or not away_team:
            continue

        game_summary = espn_client.get_game_summary(game_id)
        venue_guid = None
        if game_summary:
            game_info = game_summary.get('gameInfo', {})
            venue = game_info.get('venue', {})
            venue_guid = venue.get('guid')

        matchups.append({
            'home_team': home_team,
            'away_team': away_team,
            'game_id': game_id,
            'venue_guid': venue_guid
        })

    return matchups


def pull_matchups(year: int, month: int, day: int, league: "LeagueConfig" = None) -> List[List[str]]:
    """
    Fetch matchups for a date (simple format: list of [home, away] pairs).

    Uses ESPN API instead of web scraping.

    Args:
        year, month, day: Date to fetch matchups for
        league: Optional league config (defaults to NBA)

    Returns:
        List of [home_team, away_team] lists
    """
    from datetime import date as _date
    date_obj = _date(year, month, day)

    espn_client = ESPNClient(league=league)
    scoreboard = espn_client.get_scoreboard_site(date_obj)

    if not scoreboard:
        return []

    matchups = []
    events = scoreboard.get('events', [])

    for event in events:
        competitions = event.get('competitions', [])
        if not competitions:
            continue
        competitors = competitions[0].get('competitors', [])

        home_team = None
        away_team = None
        for comp in competitors:
            team_info = comp.get('team', {})
            abbrev = team_info.get('abbreviation', comp.get('abbreviation', '')).upper()
            if comp.get('homeAway') == 'home':
                home_team = abbrev
            else:
                away_team = abbrev

        if home_team and away_team:
            matchups.append([home_team, away_team])

    return matchups


def fetch_and_save_games(
    db,
    league_config: "LeagueConfig",
    start_date: date,
    end_date: date,
    team_only: bool = False,
    players_only: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> dict:
    """
    Fetch games from ESPN for a date range and save to MongoDB.

    Args:
        db: MongoDB database instance
        league_config: League configuration
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        team_only: Only fetch team stats
        players_only: Only fetch player stats
        dry_run: Preview without database changes
        verbose: Detailed output
        quiet: Suppress per-game/per-date output (for use with progress_callback)
        progress_callback: Optional callable(percent: int, message: str) for progress updates.
                           If provided, bypasses internal job creation and uses this callback instead.

    Returns:
        Dict with statistics (games_processed, players_processed, success, error)
    """
    espn_client = ESPNClient(league=league_config)

    # Build date list
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    total_games = 0
    total_players = 0
    total_skipped = 0

    use_threading = len(dates) > 500
    job_id = None
    # Only create internal job if no external progress_callback is provided
    if use_threading and not dry_run and not progress_callback:
        job_id = _create_job(db, league_config, len(dates), team_only, players_only)

    progress_lock = Lock()
    progress = {'days_processed': 0, 'total_days': len(dates), 'games': 0, 'players': 0, 'skipped': 0}

    def process_one_date(game_date: date) -> Tuple[int, int, int, Optional[date]]:
        if not quiet:
            print(f"\nFetching games for {game_date}...")

        scoreboard = espn_client.get_scoreboard_site(game_date)
        if not scoreboard:
            if not quiet:
                print(f"  Error fetching scoreboard for {game_date}")
            return 0, 0, 0, None

        events = scoreboard.get('events', [])
        if not events:
            if not quiet:
                print(f"  No games found for {game_date}")
            return 0, 0, 0, None

        if not quiet:
            print(f"  Found {len(events)} events from API")

        skipped_this = 0
        processed_this = 0
        players_this = 0

        for evt in events:
            gid = evt.get('id')
            if not gid:
                continue

            event_date_str = evt.get('date', '')
            short_name = evt.get('shortName', gid)

            if event_date_str:
                event_dt = _parse_espn_utc_to_eastern(event_date_str, league_config)
                if event_dt:
                    event_date = event_dt.date()
                    if event_date != game_date:
                        if not quiet:
                            print(f"    Skipping {short_name}: date mismatch (UTC: {event_date_str[:10]} -> ET: {event_date}, requested: {game_date})")
                        skipped_this += 1
                        continue
                else:
                    if not quiet:
                        print(f"    Skipping {short_name}: could not parse date {event_date_str}")
                    skipped_this += 1
                    continue

            try:
                success, pcount = _process_game(
                    gid, game_date, db, league_config, espn_client,
                    team_only=team_only, players_only=players_only,
                    dry_run=dry_run, event=evt, quiet=quiet
                )
            except Exception as e:
                if not quiet:
                    print(f"  ERROR processing game {gid} on {game_date}: {e}")
                if job_id:
                    _record_failure(db, league_config, job_id, str(gid), str(e))
                continue

            if success:
                processed_this += 1
                players_this += pcount

        if not quiet:
            print(f"  Processed: {processed_this}, Skipped (wrong date): {skipped_this}")
        return processed_this, players_this, skipped_this, game_date

    if use_threading:
        chunks = [dates[i:i + 50] for i in range(0, len(dates), 50)]
        max_workers = min(10, len(chunks))

        def process_chunk(chunk_dates, chunk_idx):
            cg, cp, cs = 0, 0, 0
            for d in chunk_dates:
                g, p, s, _ = process_one_date(d)
                cg += g
                cp += p
                cs += s
                with progress_lock:
                    progress['days_processed'] += 1
                    progress['games'] += g
                    progress['players'] += p
                    progress['skipped'] += s
                    days_done = progress['days_processed']
                    games_done = progress['games']
                # Report progress via callback or internal job
                should_report = days_done % 5 == 0 or days_done == progress['total_days']
                if should_report:
                    pct = int(100 * days_done / progress['total_days']) if progress['total_days'] > 0 else 0
                    msg = f"Pulled {games_done} games ({days_done}/{progress['total_days']} days)"
                    if progress_callback:
                        progress_callback(pct, msg)
                    elif job_id:
                        _update_job(db, league_config, job_id, days_done, progress['total_days'],
                                    games_done, progress['players'], progress['skipped'])
            return cg, cp, cs

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_chunk, chunk, idx + 1): idx
                    for idx, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    try:
                        g, p, s = future.result()
                        total_games += g
                        total_players += p
                        total_skipped += s
                    except Exception as e:
                        print(f"  ERROR in date-chunk {futures[future] + 1}: {e}")

            if job_id:
                with progress_lock:
                    _complete_job(db, league_config, job_id,
                                  progress['days_processed'], progress['total_days'],
                                  progress['games'], progress['players'], progress['skipped'])
        except Exception as e:
            if job_id:
                _fail_job(db, league_config, job_id, str(e))
            raise
    else:
        total_days = len(dates)
        for idx, game_date in enumerate(dates):
            g, p, s, _ = process_one_date(game_date)
            total_games += g
            total_players += p
            total_skipped += s
            # Report progress via callback for non-threading path
            if progress_callback:
                days_done = idx + 1
                pct = int(100 * days_done / total_days) if total_days > 0 else 0
                msg = f"Pulled {total_games} games ({days_done}/{total_days} days)"
                progress_callback(pct, msg)

    if not quiet:
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
        print(f"  Games processed: {total_games}")
        print(f"  Games skipped (date mismatch): {total_skipped}")
        if not team_only:
            print(f"  Player stats processed: {total_players}")

    return {
        'games_processed': total_games,
        'players_processed': total_players,
        'success': True,
        'error': None
    }


def refresh_venues(
    db,
    league_config: "LeagueConfig",
    dry_run: bool = False
) -> dict:
    """
    Refresh venues collection with location data from league venues JSON file.

    Args:
        db: MongoDB database instance
        league_config: League configuration
        dry_run: Preview without database changes

    Returns:
        Dict with statistics
    """
    from bball.data.league_db_proxy import LeagueDbProxy
    league_db = LeagueDbProxy(db, league_config)

    print("Refreshing venue location data...")

    # Look for venues JSON data file
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Search multiple locations for the venues JSON
    _legacy_dir = os.path.join(project_root, 'cli' + '_old')
    _data_dir = os.path.join(project_root, 'data')
    candidates = [
        os.path.join(_data_dir, f'{league_config.league_id}_venues.json'),
        os.path.join(_data_dir, 'nba_venues.json'),
        os.path.join(_legacy_dir, f'{league_config.league_id}_venues.json'),
        os.path.join(_legacy_dir, 'nba_venues.json'),
    ]
    json_path = None
    for candidate in candidates:
        if os.path.exists(candidate):
            json_path = candidate
            break

    if json_path is None:
        print(f"  Error: venues JSON not found in any search path")
        return {'success': False, 'error': 'venues JSON not found'}

    with open(json_path, 'r') as f:
        venue_locations = json.load(f)

    print(f"  Loaded {len(venue_locations)} venues from JSON")

    matched_count = 0
    unmatched_count = 0
    matched_venues = []
    unmatched_venues = []

    db_venues = list(league_db.nba_venues.find({}, {'venue_guid': 1, 'fullName': 1}))
    print(f"  Found {len(db_venues)} venues in database")

    location_lookup = {v['fullName']: v for v in venue_locations}

    for db_venue in db_venues:
        full_name = db_venue.get('fullName')
        venue_guid = db_venue.get('venue_guid')
        if not full_name or not venue_guid:
            continue

        location_data = location_lookup.get(full_name)
        if location_data and 'lat' in location_data and 'lon' in location_data:
            location = {'lat': location_data['lat'], 'lon': location_data['lon']}
            if not dry_run:
                league_db.nba_venues.update_one(
                    {'venue_guid': venue_guid},
                    {'$set': {'location': location}}
                )
            matched_count += 1
            matched_venues.append(full_name)
        else:
            unmatched_count += 1
            unmatched_venues.append(full_name)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Results:")
    print(f"  Matched: {matched_count}")
    if matched_venues:
        print(f"    Venues matched:")
        for v in matched_venues[:10]:
            print(f"      - {v}")
        if len(matched_venues) > 10:
            print(f"      ... and {len(matched_venues) - 10} more")
    print(f"  Unmatched: {unmatched_count}")
    if unmatched_venues:
        print(f"    Venues not matched:")
        for v in unmatched_venues[:10]:
            print(f"      - {v}")
        if len(unmatched_venues) > 10:
            print(f"      ... and {len(unmatched_venues) - 10} more")

    return {'success': True, 'matched': matched_count, 'unmatched': unmatched_count}


def geocode_missing_venues(db, league_config, dry_run=False):
    """
    Geocode venues that are missing location coordinates.

    Uses Nominatim geocoding with ESPN address fields (fullName, city, state).
    Rate-limited to 1 request/second per Nominatim ToS.

    Args:
        db: MongoDB database instance
        league_config: League configuration
        dry_run: Preview without database changes

    Returns:
        Dict with statistics: geocoded, failed, skipped, already_have_location
    """
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError

    from bball.data.league_db_proxy import LeagueDbProxy
    league_db = LeagueDbProxy(db, league_config)

    venues_col = league_db.nba_venues

    already_have = venues_col.count_documents({'location.lat': {'$exists': True}})
    missing = list(venues_col.find(
        {'$or': [
            {'location': {'$exists': False}},
            {'location.lat': {'$exists': False}},
        ]},
        {'venue_guid': 1, 'fullName': 1, 'address': 1}
    ))

    if not missing:
        return {'geocoded': 0, 'failed': 0, 'skipped': 0, 'already_have_location': already_have}

    geolocator = Nominatim(user_agent="bball/1.0", timeout=10)

    geocoded = 0
    failed = 0
    skipped = 0

    for i, venue in enumerate(missing):
        venue_guid = venue.get('venue_guid')
        full_name = venue.get('fullName', '')
        address = venue.get('address') or {}

        if not full_name:
            skipped += 1
            continue

        city = address.get('city', '') if isinstance(address, dict) else ''
        state = address.get('state', '') if isinstance(address, dict) else ''

        # Primary query: arena name + city + state
        query_parts = [full_name]
        if city:
            query_parts.append(city)
        if state:
            query_parts.append(state)
        query = ", ".join(query_parts)

        # Rate limit: 1 req/sec
        if i > 0:
            time.sleep(1)

        try:
            location = geolocator.geocode(query)

            # Fallback: try street address if arena name didn't work
            if not location and address.get('street'):
                fallback_parts = [address['street']]
                if city:
                    fallback_parts.append(city)
                if state:
                    postal = address.get('postalCode', '')
                    fallback_parts.append(f"{state} {postal}".strip())
                fallback_query = ", ".join(fallback_parts)
                time.sleep(1)
                location = geolocator.geocode(fallback_query)

            if location:
                if not dry_run:
                    venues_col.update_one(
                        {'venue_guid': venue_guid},
                        {'$set': {'location': {'lat': location.latitude, 'lon': location.longitude}}}
                    )
                geocoded += 1
            else:
                failed += 1
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"    Warning: Geocoding failed for '{full_name}': {e}")
            failed += 1
        except Exception as e:
            print(f"    Warning: Unexpected geocoding error for '{full_name}': {e}")
            failed += 1

    print(f"  Geocoded {geocoded}/{len(missing)} venues ({failed} failed, {skipped} skipped)")

    return {
        'geocoded': geocoded,
        'failed': failed,
        'skipped': skipped,
        'already_have_location': already_have,
    }


def refresh_players(
    db,
    league_config: "LeagueConfig",
    dry_run: bool = False
) -> dict:
    """
    Refresh players collection with metadata from player_stats collection.

    Args:
        db: MongoDB database instance
        league_config: League configuration
        dry_run: Preview without database changes

    Returns:
        Dict with statistics
    """
    from bball.data.league_db_proxy import LeagueDbProxy
    league_db = LeagueDbProxy(db, league_config)

    print("Refreshing players metadata...")

    pipeline = [
        {
            '$group': {
                '_id': '$player_id',
                'player_name': {'$first': '$player_name'},
                'short_name': {'$first': '$short_name'},
                'guid': {'$first': '$guid'},
                'last_game_date': {'$max': '$date'},
            }
        }
    ]

    players = list(league_db.stats_nba_players.aggregate(pipeline))

    if dry_run:
        print(f"  [DRY RUN] Would upsert {len(players)} players")
        for p in players[:5]:
            print(f"    - {p.get('player_name')} (ID: {p.get('_id')})")
        if len(players) > 5:
            print(f"    ... and {len(players) - 5} more")
    else:
        for player in players:
            player_name = player.get('player_name', '')
            player_slug = player_name.lower().replace(' ', '-').replace("'", '').replace('.', '')
            espn_link = f"https://www.espn.com/{league_config.league_id}/player/_/id/{player['_id']}/{player_slug}"

            player_doc = {
                'player_id': player['_id'],
                'player_name': player.get('player_name'),
                'short_name': player.get('short_name'),
                'guid': player.get('guid'),
                'last_game_date': player.get('last_game_date'),
                'espn_link': espn_link,
                'updated_at': datetime.now()
            }

            league_db.players_nba.update_one(
                {'player_id': player['_id']},
                {'$set': player_doc, '$setOnInsert': {'created_at': datetime.now()}},
                upsert=True
            )

        print(f"  Upserted {len(players)} players")

    return {'success': True, 'players_count': len(players)}


def refresh_teams(
    db,
    league_config: "LeagueConfig",
    dry_run: bool = False
) -> dict:
    """
    Refresh teams collection from ESPN teams API.

    Fetches all teams (with pagination), extracts conference info,
    and upserts into the league's teams collection.

    Args:
        db: MongoDB database instance
        league_config: League configuration
        dry_run: Preview without database changes

    Returns:
        Dict with statistics
    """
    from bball.data.espn_client import ESPNClient
    from bball.data.league_db_proxy import LeagueDbProxy

    league_db = LeagueDbProxy(db, league_config)
    client = ESPNClient(league=league_config)

    print("Refreshing teams metadata from ESPN...")

    # Fetch all teams with pagination
    all_teams = []
    page = 1
    while True:
        data = client.get_teams(page=page, limit=500)
        if not data:
            break

        page_teams = []

        # ESPN response: {sports: [{leagues: [{teams: [...]}]}]}
        for sport in data.get('sports', []):
            for league_obj in sport.get('leagues', []):
                for team_wrapper in league_obj.get('teams', []):
                    team = team_wrapper.get('team', team_wrapper)
                    page_teams.append(team)

        # Flat structure fallback
        if not page_teams:
            for team_wrapper in data.get('teams', []):
                team = team_wrapper.get('team', team_wrapper)
                page_teams.append(team)

        if not page_teams:
            break

        all_teams.extend(page_teams)

        page_count = data.get('pageCount', 1)
        if page >= page_count:
            break
        page += 1

    if not all_teams:
        print("  No teams fetched from ESPN API")
        return {'success': False, 'error': 'No teams from API'}

    print(f"  Fetched {len(all_teams)} teams from ESPN ({page} page(s))")

    if dry_run:
        print(f"  [DRY RUN] Would upsert {len(all_teams)} teams")
        for t in all_teams[:5]:
            print(f"    - {t.get('displayName')} ({t.get('abbreviation')})")
        if len(all_teams) > 5:
            print(f"    ... and {len(all_teams) - 5} more")
        return {'success': True, 'teams_count': len(all_teams)}

    upserted = 0
    for team in all_teams:
        team_id = str(team.get('id', ''))
        if not team_id:
            continue

        # Extract conference from groups
        conference = None
        conference_id = None
        groups = team.get('groups', {})
        if isinstance(groups, dict):
            parent = groups.get('parent', {})
            if parent and parent.get('name'):
                conference = parent['name']
                conference_id = str(parent.get('id', ''))
            elif groups.get('name'):
                conference = groups['name']
                conference_id = str(groups.get('id', ''))
        elif isinstance(groups, list):
            for g in groups:
                if g.get('isConference') or g.get('type') == 'conference':
                    conference = g.get('name') or g.get('shortName')
                    conference_id = str(g.get('id', ''))
                    break
            if not conference and groups:
                conference = groups[0].get('name')
                conference_id = str(groups[0].get('id', ''))

        # Extract logo URL
        logo = None
        logos = team.get('logos', [])
        if logos and isinstance(logos, list):
            logo = logos[0].get('href', '')
        if not logo:
            logo = f"https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png"

        team_doc = {
            'team_id': team_id,
            'id': team_id,
            'abbreviation': team.get('abbreviation', ''),
            'displayName': team.get('displayName', ''),
            'shortDisplayName': team.get('shortDisplayName', ''),
            'name': team.get('name', ''),
            'location': team.get('location', ''),
            'slug': team.get('slug', ''),
            'uid': team.get('uid', ''),
            'color': team.get('color', ''),
            'alternateColor': team.get('alternateColor', ''),
            'logo': logo,
            'links': team.get('links', []),
            'nickname': team.get('nickname', ''),
            'isActive': team.get('isActive', True),
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
        }

        if conference:
            team_doc['conference'] = conference
        if conference_id:
            team_doc['conference_id'] = conference_id

        league_db.teams_nba.update_one(
            {'team_id': team_id},
            {'$set': team_doc, '$setOnInsert': {'created_at': datetime.now()}},
            upsert=True
        )
        upserted += 1

    print(f"  Upserted {upserted} teams")

    return {'success': True, 'teams_count': upserted}


def refresh_team_conferences(
    db,
    league_config: "LeagueConfig",
    dry_run: bool = False
) -> dict:
    """
    Refresh team conference data from ESPN groups endpoint.

    The /teams endpoint doesn't include conference info for college sports.
    The /groups endpoint returns conferences with their member teams, which
    is the only reliable source for this data.

    Args:
        db: MongoDB database instance
        league_config: League configuration
        dry_run: Preview without database changes

    Returns:
        Dict with statistics: success, updated, conferences
    """
    from bball.data.league_db_proxy import LeagueDbProxy

    league_db = LeagueDbProxy(db, league_config)
    espn = league_config.espn
    base_url_site = espn.get('base_url_site', 'https://site.api.espn.com')
    sport_path = espn.get('sport_path', league_config.league_id)

    url = f"{base_url_site}/apis/site/v2/sports/basketball/{sport_path}/groups"

    print(f"Fetching conference data from ESPN groups endpoint...")

    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  Error fetching groups: {e}")
        return {'success': False, 'error': str(e), 'updated': 0, 'conferences': 0}

    # Parse conferences and their teams
    # ESPN groups response can be:
    #   { "groups": [{ "children": [{ "name": "ACC", "teams": [...] }] }] }
    # or sometimes groups at the top level ARE the conferences with teams directly
    conf_count = 0
    updated = 0

    top_groups = data.get('groups', [])
    if not top_groups:
        top_groups = data.get('children', [])

    conferences = []
    for group in top_groups:
        children = group.get('children', [])
        if children:
            # Two-level: divisions -> conferences with teams
            conferences.extend(children)
        elif group.get('teams'):
            # Single-level: group itself is a conference with teams
            conferences.append(group)

    if not conferences:
        print(f"  No conferences found in groups response (keys: {list(data.keys())})")
        return {'success': False, 'error': 'No conferences in response', 'updated': 0, 'conferences': 0}

    for conf in conferences:
        conf_name = conf.get('name') or conf.get('shortName') or conf.get('abbreviation')
        if not conf_name:
            continue

        conf_count += 1
        teams = conf.get('teams', [])

        for team_wrapper in teams:
            team = team_wrapper.get('team', team_wrapper)
            team_id = str(team.get('id', ''))
            if not team_id:
                continue

            if dry_run:
                abbrev = team.get('abbreviation', '?')
                print(f"  [DRY RUN] {abbrev} (id={team_id}) -> {conf_name}")
                updated += 1
                continue

            result = league_db.teams_nba.update_one(
                {'team_id': team_id},
                {'$set': {'conference': conf_name}},
            )
            if result.matched_count > 0:
                updated += 1

    print(f"  {'[DRY RUN] ' if dry_run else ''}Found {conf_count} conferences, updated {updated} teams")

    return {'success': True, 'updated': updated, 'conferences': conf_count}


def _fetch_scoreboard_with_retry(espn_client, game_date, max_retries=3, base_timeout=60):
    """Fetch scoreboard with retry + exponential backoff for heavy dates (CBB)."""
    for attempt in range(max_retries):
        try:
            # Build URL manually with longer timeout
            date_str = game_date.strftime('%Y%m%d')
            url = espn_client._url("scoreboard_site_template", YYYYMMDD=date_str)
            timeout = base_timeout * (attempt + 1)
            response = requests.get(url, headers=espn_client.headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                print(f"  Failed after {max_retries} retries for {game_date}: {e}")
                return None
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (502, 503, 504):
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    time.sleep(wait)
                else:
                    print(f"  Failed after {max_retries} retries for {game_date}: {e}")
                    return None
            else:
                print(f"  HTTP error for {game_date}: {e}")
                return None
        except requests.RequestException as e:
            print(f"  Request error for {game_date}: {e}")
            return None
    return None


def backfill_field_from_scoreboard(
    db,
    league_config: "LeagueConfig",
    field_name: str,
    extractor=None,
    seasons: Optional[List[str]] = None,
    only_missing: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Backfill a field on existing game documents from ESPN scoreboard API.

    Uses the scoreboard (one API call per date) instead of game summaries
    (one call per game), so it's efficient for bulk backfill. Includes retry
    with exponential backoff for heavy dates (CBB can have 100+ games/day).

    Args:
        db: MongoDB database instance
        league_config: League configuration
        field_name: Field name to set on game documents
        extractor: Optional callable(competition_dict) -> value.
                   If None, uses competition.get(field_name).
        seasons: Optional list of seasons to filter (e.g., ['2024-2025']).
                 If None, backfills all games.
        only_missing: If True, only backfill games that don't already have
                      the field. Useful for resuming interrupted backfills.
        dry_run: Preview without database changes

    Returns:
        Dict with {dates_processed, games_updated, games_skipped, errors}

    Usage:
        # Backfill neutralSite for all games
        backfill_field_from_scoreboard(db, league, 'neutralSite')

        # Backfill for specific seasons
        backfill_field_from_scoreboard(db, league, 'neutralSite', seasons=['2024-2025'])

        # Custom extractor
        backfill_field_from_scoreboard(db, league, 'conferenceCompetition',
            extractor=lambda comp: bool(comp.get('conferenceCompetition', False)))
    """
    from bball.data.league_db_proxy import LeagueDbProxy
    league_db = LeagueDbProxy(db, league_config)

    if extractor is None:
        extractor = lambda comp: comp.get(field_name)

    # Get distinct dates from games collection
    query = {}
    if seasons:
        query['season'] = {'$in': seasons}
    if only_missing:
        query[field_name] = {'$exists': False}

    dates_cursor = league_db.stats_nba.distinct('date', query)
    all_dates = sorted(set(d for d in dates_cursor if d))

    print(f"Backfilling '{field_name}' for {len(all_dates)} dates"
          f"{f' (seasons: {seasons})' if seasons else ''}"
          f"{' [DRY RUN]' if dry_run else ''}...")

    espn_client = ESPNClient(league=league_config)
    dates_processed = 0
    games_updated = 0
    games_skipped = 0
    errors = []

    for date_str in all_dates:
        try:
            game_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            errors.append(f"Invalid date: {date_str}")
            continue

        scoreboard = _fetch_scoreboard_with_retry(espn_client, game_date)
        if not scoreboard:
            errors.append(f"No scoreboard for {date_str}")
            continue

        # Build game_id -> extracted value from scoreboard
        values_by_game = {}
        for event in scoreboard.get('events', []):
            game_id = event.get('id')
            if not game_id:
                continue
            competitions = event.get('competitions', [])
            if not competitions:
                continue
            value = extractor(competitions[0])
            if value is not None:
                values_by_game[str(game_id)] = value

        # Get game_ids in our DB for this date
        date_query = {'date': date_str}
        if only_missing:
            date_query[field_name] = {'$exists': False}
        db_games = league_db.stats_nba.find(date_query, {'game_id': 1})

        for doc in db_games:
            gid = str(doc.get('game_id'))
            if gid in values_by_game:
                if not dry_run:
                    league_db.stats_nba.update_one(
                        {'game_id': gid},
                        {'$set': {field_name: values_by_game[gid]}}
                    )
                games_updated += 1
            else:
                games_skipped += 1

        dates_processed += 1
        if dates_processed % 50 == 0:
            print(f"  Progress: {dates_processed}/{len(all_dates)} dates, "
                  f"{games_updated} games updated")

    print(f"{'[DRY RUN] ' if dry_run else ''}Done: {dates_processed} dates, "
          f"{games_updated} games updated, {games_skipped} skipped, "
          f"{len(errors)} errors")

    return {
        'dates_processed': dates_processed,
        'games_updated': games_updated,
        'games_skipped': games_skipped,
        'errors': errors,
    }
