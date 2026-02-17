"""
Roster Service - Builds team rosters from player game stats.

Analyzes stats_nba_players (or league-equivalent) to determine each team's
roster and likely starters for a given season.

Starter selection logic:
- Top 2 guards by games started
- Top 2 forwards by games started
- Top 1 center by games started

Players with < 5 MPG or 0 total minutes are excluded from rosters.
A player's current team is their most recent game's team.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, TYPE_CHECKING

from bball_app.core.data.rosters import RostersRepository

if TYPE_CHECKING:
    from bball_app.core.league_config import LeagueConfig

logger = logging.getLogger(__name__)

MIN_MPG = 5.0
STARTER_SLOTS = {'guard': 2, 'forward': 2, 'center': 1}


def _get_position_category(pos_display_name: Optional[str]) -> Optional[str]:
    """Map pos_display_name ('Guard', 'Forward', 'Center') to a category key."""
    if not pos_display_name:
        return None
    key = pos_display_name.strip().lower()
    if key in ('guard', 'forward', 'center'):
        return key
    return None


def _gather_player_season_data(db, season: str, player_stats_collection: str) -> Dict:
    """
    Query every player-game record for *season* with minutes > 0.

    Returns a dict keyed by player_id with aggregated totals, deduped by
    game_id (handles duplicate records for the same game gracefully).
    """
    cursor = db[player_stats_collection].find(
        {'season': season, 'stats.min': {'$gt': 0}},
        {
            'player_id': 1,
            'team': 1,
            'date': 1,
            'game_id': 1,
            'stats.min': 1,
            'starter': 1,
            'player_name': 1,
        },
    ).sort('date', 1)

    players: Dict[str, dict] = {}
    for pg in cursor:
        pid = str(pg.get('player_id', ''))
        gid = pg.get('game_id')
        if not pid or not gid:
            continue

        if pid not in players:
            players[pid] = {
                'player_id': pid,
                'player_name': pg.get('player_name', ''),
                'last_team': None,
                'game_data': {},
            }

        # Sorted ascending by date — last write wins for team assignment.
        players[pid]['last_team'] = pg.get('team')
        players[pid]['game_data'][gid] = {
            'min': pg.get('stats', {}).get('min', 0),
            'starter': pg.get('starter', False),
        }

    # Derive aggregates
    for data in players.values():
        gd = data['game_data']
        data['total_games'] = len(gd)
        data['games_started'] = sum(1 for g in gd.values() if g.get('starter'))
        data['total_minutes'] = sum(g.get('min', 0) for g in gd.values())
        data['mpg'] = data['total_minutes'] / data['total_games'] if data['total_games'] else 0.0

    return players


def _lookup_positions(db, player_ids: List[str], players_collection: str) -> Dict[str, Optional[str]]:
    """Batch-fetch pos_display_name from the players collection."""
    if not player_ids:
        return {}
    cursor = db[players_collection].find(
        {'player_id': {'$in': player_ids}},
        {'player_id': 1, 'pos_display_name': 1},
    )
    return {str(doc.get('player_id', '')): doc.get('pos_display_name') for doc in cursor}


def _select_starters(players_on_team: List[dict]) -> set:
    """Pick starters from a single team's player list using positional rules."""
    by_pos: Dict[str, List[dict]] = defaultdict(list)
    for p in players_on_team:
        cat = p.get('position_category')
        if cat:
            by_pos[cat].append(p)

    for bucket in by_pos.values():
        bucket.sort(key=lambda x: x['games_started'], reverse=True)

    starter_ids: set = set()
    for pos, slots in STARTER_SLOTS.items():
        for p in by_pos.get(pos, [])[:slots]:
            starter_ids.add(p['player_id'])
    return starter_ids


def build_rosters(
    db,
    season: str,
    league: Optional["LeagueConfig"] = None,
    *,
    dry_run: bool = False,
) -> int:
    """
    Build and upsert team rosters for *season* from player game data.

    Parameters
    ----------
    db : pymongo database
    season : e.g. "2024-2025"
    league : LeagueConfig (uses collection names; falls back to NBA defaults)
    dry_run : log what would change without writing to MongoDB

    Returns
    -------
    int : number of team rosters upserted
    """
    if league is not None:
        player_stats_col = league.collections.get('player_stats', 'nba_player_stats')
        players_col = league.collections.get('players', 'nba_players')
    else:
        player_stats_col = 'nba_player_stats'
        players_col = 'nba_players'

    logger.info("Building rosters for season %s (player_stats=%s, players=%s)",
                season, player_stats_col, players_col)

    # 1. Gather per-player aggregates
    all_players = _gather_player_season_data(db, season, player_stats_col)
    logger.info("Found %d players with minutes > 0 in season %s", len(all_players), season)

    # 2. Filter to >= MIN_MPG and look up positions
    eligible_ids = [pid for pid, d in all_players.items() if d['mpg'] >= MIN_MPG]
    positions = _lookup_positions(db, eligible_ids, players_col)

    # 3. Group by team
    teams: Dict[str, List[dict]] = defaultdict(list)
    for pid in eligible_ids:
        d = all_players[pid]
        team = d.get('last_team')
        if not team:
            continue
        pos_name = positions.get(pid)
        teams[team].append({
            'player_id': pid,
            'player_name': d.get('player_name', ''),
            'position': pos_name,
            'position_category': _get_position_category(pos_name),
            'games_started': d['games_started'],
            'total_minutes': d['total_minutes'],
            'total_games': d['total_games'],
        })

    logger.info("Players distributed across %d teams", len(teams))

    # 4. Select starters and upsert
    repo = RostersRepository(db, league=league)
    updated = 0

    for team, players_on_team in teams.items():
        starter_ids = _select_starters(players_on_team)
        roster = [
            {'player_id': p['player_id'], 'starter': p['player_id'] in starter_ids}
            for p in players_on_team
        ]
        starters_count = sum(1 for r in roster if r['starter'])

        if dry_run:
            starter_names = [p['player_name'] for p in players_on_team if p['player_id'] in starter_ids]
            logger.info("[DRY RUN] %s: %d players, %d starters (%s)",
                        team, len(roster), starters_count, ', '.join(starter_names[:5]))
        else:
            repo.upsert_roster(team, season, roster)
            logger.info("%s: %d players, %d starters", team, len(roster), starters_count)

        updated += 1

    logger.info("%s %d team rosters for season %s",
                "Would update" if dry_run else "Updated", updated, season)
    return updated
