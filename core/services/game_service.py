"""
Game Service - Core logic for game detail operations.

Provides game detail data including rosters, predictions, and team info.
League-aware through LeagueConfig.
"""

from datetime import date, datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from bball_app.core.utils import get_season_from_date

if TYPE_CHECKING:
    from bball_app.core.league_config import LeagueConfig


def get_position_sort_order(pos_name: str) -> int:
    """Get sort order for position (guard=0, forward=1, center=2)."""
    if not pos_name:
        return 3
    pos_lower = pos_name.lower()
    if 'guard' in pos_lower or pos_lower in ('pg', 'sg', 'g'):
        return 0
    elif 'forward' in pos_lower or pos_lower in ('sf', 'pf', 'f'):
        return 1
    elif 'center' in pos_lower or pos_lower in ('c',):
        return 2
    return 3


def calculate_player_stats(db, player_id: str, team: str, season: str, before_date: str,
                          player_stats_collection: str = 'nba_player_stats',
                          exclude_game_types: List[str] = None) -> Dict:
    """Calculate player stats for display."""
    # Handle both string and int player_ids
    player_id_query = [player_id]
    if player_id.isdigit():
        player_id_query.append(int(player_id))

    # Build match query - don't filter by team to catch traded players
    # Only count games where player actually played (stats.min exists and > 0)
    match_query = {
        'player_id': {'$in': player_id_query},
        'season': season,
        'date': {'$lte': before_date},  # Include games up to and including the date
        'stats.min': {'$exists': True, '$gt': 0}  # Only games where player had minutes
    }

    # Exclude preseason/allstar if specified
    if exclude_game_types:
        match_query['game_type'] = {'$nin': exclude_game_types}

    # Stats are nested under 'stats' object in player_stats documents
    pipeline = [
        {'$match': match_query},
        {
            '$group': {
                '_id': None,
                'games': {'$sum': 1},
                'games_started': {
                    '$sum': {'$cond': [{'$eq': ['$starter', True]}, 1, 0]}
                },
                'total_pts': {'$sum': {'$toDouble': {'$ifNull': ['$stats.pts', 0]}}},
                'total_reb': {'$sum': {'$toDouble': {'$ifNull': ['$stats.reb', 0]}}},
                'total_ast': {'$sum': {'$toDouble': {'$ifNull': ['$stats.ast', 0]}}},
                'total_min': {'$sum': {'$toDouble': {'$ifNull': ['$stats.min', 0]}}}
            }
        }
    ]

    result = list(db[player_stats_collection].aggregate(pipeline))

    if not result or result[0]['games'] == 0:
        return {'ppg': 0.0, 'rpg': 0.0, 'apg': 0.0, 'mpg': 0.0, 'games': 0, 'games_started': 0, 'min': 0}

    stats = result[0]
    games = stats['games']

    return {
        'ppg': round(stats['total_pts'] / games, 1) if games > 0 else 0.0,
        'rpg': round(stats['total_reb'] / games, 1) if games > 0 else 0.0,
        'apg': round(stats['total_ast'] / games, 1) if games > 0 else 0.0,
        'mpg': round(stats['total_min'] / games, 1) if games > 0 else 0.0,
        'games': games,
        'games_started': stats.get('games_started', 0),
        'min': stats['total_min']
    }


def get_team_players(db, team: str, season: str, game_date: date,
                     league: "LeagueConfig") -> List[Dict]:
    """
    Get players for a team from the rosters collection with stats.

    Players are pulled directly from the rosters collection, with additional
    details (headshot, position) enriched from the players collection.
    The starter/injured flags come from the roster entries.

    Args:
        db: MongoDB database instance
        team: Team identifier (abbreviation for NBA, team_id for CBB)
        season: Season string (e.g., '2024-2025')
        game_date: Date of the game
        league: LeagueConfig for collection names

    Returns:
        List of player dictionaries with roster info and stats
    """
    rosters_collection = league.collections.get('rosters', 'nba_rosters')
    players_collection = league.collections.get('players', 'nba_players')
    player_stats_collection = league.collections.get('player_stats', 'nba_player_stats')

    before_date = game_date.strftime('%Y-%m-%d')
    result = []

    # Get roster from rosters collection - this is the source of truth for who's on the team
    roster_doc = db[rosters_collection].find_one({'season': season, 'team': team})

    if not roster_doc or not roster_doc.get('roster'):
        # No roster found - return empty list
        # Could fall back to players collection, but roster is the source of truth
        return []

    roster = roster_doc.get('roster', [])

    # Build a map of player details from players collection for enrichment
    player_ids = [str(p.get('player_id', '')) for p in roster if p.get('player_id')]

    # Query players collection for additional details (headshot, position)
    expanded_ids = []
    for pid in player_ids:
        expanded_ids.append(pid)
        if pid.isdigit():
            expanded_ids.append(int(pid))

    players_details = {}
    if expanded_ids:
        players_cursor = db[players_collection].find(
            {'player_id': {'$in': expanded_ids}},
            {
                'player_id': 1, 'player_name': 1, 'headshot': 1,
                'pos_name': 1, 'pos_display_name': 1,
                'injury_status': 1
            }
        )
        for p in players_cursor:
            players_details[str(p.get('player_id', ''))] = p

    # Iterate over roster entries - roster is the source of truth
    for roster_entry in roster:
        player_id = str(roster_entry.get('player_id', ''))
        if not player_id:
            continue

        # Get player details from players collection (for headshot, position)
        player_doc = players_details.get(player_id, {})

        # Player name: prefer roster entry, fall back to players collection
        player_name = roster_entry.get('player_name') or player_doc.get('player_name', '')
        if not player_name or not player_name.strip():
            continue

        # Get starter flag from roster - check both naming conventions
        # New convention: 'starter', Legacy: 'is_starter'
        is_starter = roster_entry.get('starter', roster_entry.get('is_starter', False))

        # Get injured flag from roster - check both naming conventions
        # New convention: 'injured', Legacy: 'is_playing' (False means injured/out)
        if 'injured' in roster_entry:
            is_injured = roster_entry.get('injured', False)
        elif 'is_playing' in roster_entry:
            is_injured = not roster_entry.get('is_playing', True)
        else:
            is_injured = False

        # Get disabled flag from roster (user-toggled exclusion from predictions)
        is_disabled = roster_entry.get('disabled', False)

        # Calculate stats (exclude preseason/allstar based on league config)
        exclude_game_types = league.exclude_game_types if hasattr(league, 'exclude_game_types') else []
        stats = calculate_player_stats(db, player_id, team, season, before_date, player_stats_collection, exclude_game_types)

        # Headshot and position always come from players collection
        headshot = player_doc.get('headshot')
        pos_name = player_doc.get('pos_name')
        pos_display_name = player_doc.get('pos_display_name')

        result.append({
            'player_id': player_id,
            'player_name': player_name,
            'headshot': headshot,
            'pos_name': pos_name,
            'pos_display_name': pos_display_name,
            'injured': is_injured,
            'injury_status': player_doc.get('injury_status'),
            'starter': is_starter,
            'disabled': is_disabled,
            'stats': stats
        })

    # Sort: starters first, then by position, then by name
    result.sort(key=lambda p: (
        not p.get('starter', False),
        p.get('injured', False),  # Non-injured before injured within each group
        get_position_sort_order(p.get('pos_name', '')),
        p.get('player_name', '')
    ))

    return result


def get_team_info(db, team: str, league: "LeagueConfig") -> Dict:
    """Get team logo and colors from teams collection."""
    teams_collection = league.collections.get('teams', 'nba_teams')

    # Query by the league's primary identifier field
    # NBA uses abbreviation, CBB uses team_id
    if league.team_primary_identifier == 'id':
        # CBB: team is the ESPN team_id, but fall back to abbreviation lookup
        # since game docs may only have the abbreviation (stored as 'name')
        team_data = db[teams_collection].find_one({'id': team}) or \
                    db[teams_collection].find_one({'team_id': team}) or \
                    db[teams_collection].find_one({'id': str(team)}) or \
                    db[teams_collection].find_one({'abbreviation': team}) or {}
    else:
        # NBA: team is the abbreviation
        team_data = db[teams_collection].find_one({'abbreviation': team}) or {}

    logo = team_data.get('logo')
    if not logo:
        logos = team_data.get('logos', [])
        if isinstance(logos, list) and len(logos) > 0:
            first_logo = logos[0]
            if isinstance(first_logo, dict) and first_logo.get('href'):
                logo = first_logo['href']

    return {
        'logo': logo,
        'color': team_data.get('color', '667eea'),
        'alternate_color': team_data.get('alternateColor', '764ba2'),
        'name': team_data.get('abbreviation') or team_data.get('name', team)
    }


def get_game_detail(db, game_id: str, game_date: Optional[date],
                    league: "LeagueConfig") -> Dict:
    """
    Get complete game detail data for modal display.

    Args:
        db: MongoDB database instance
        game_id: ESPN game ID
        game_date: Optional date (will be fetched from DB if not provided)
        league: LeagueConfig for collection names and season logic

    Returns:
        Dictionary with game detail data or error
    """
    games_collection = league.collections.get('games', 'stats_nba')
    game = db[games_collection].find_one({'game_id': game_id})

    if not game:
        return {'success': False, 'error': 'Game not found'}

    # Get team identifier based on league config (NBA uses 'name'/abbreviation, CBB uses 'id')
    # Fall back to 'team_id' then 'name' if the primary identifier field is not present
    team_id_field = league.team_primary_identifier  # 'name' or 'id'
    home_team = game.get('homeTeam', {}).get(team_id_field, '')
    away_team = game.get('awayTeam', {}).get(team_id_field, '')

    # Fallback: CBB game docs store 'team_id' not 'id', so try that next
    if not home_team:
        home_team = game.get('homeTeam', {}).get('team_id', '')
    if not away_team:
        away_team = game.get('awayTeam', {}).get('team_id', '')

    # Final fallback to 'name' (abbreviation) if no numeric id found
    if not home_team:
        home_team = game.get('homeTeam', {}).get('name', '')
    if not away_team:
        away_team = game.get('awayTeam', {}).get('name', '')

    # Also get display names for UI
    home_team_display = game.get('homeTeam', {}).get('name', home_team)
    away_team_display = game.get('awayTeam', {}).get('name', away_team)

    if not home_team or not away_team:
        return {'success': False, 'error': 'Game missing team information'}

    # Get game date
    if not game_date:
        game_date_str = game.get('date')
        if game_date_str:
            try:
                game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date()
            except:
                game_date = date.today()
        else:
            game_date = date.today()

    # Get season
    season = get_season_from_date(game_date, league=league)

    # Get players for both teams
    home_players = get_team_players(db, home_team, season, game_date, league)
    away_players = get_team_players(db, away_team, season, game_date, league)

    # Get team info
    home_team_info = get_team_info(db, home_team, league)
    away_team_info = get_team_info(db, away_team, league)

    # Get team stats (W-L records)
    game_date_str = game_date.strftime('%Y-%m-%d')
    home_team_stats = get_team_stats(db, home_team, season, game_date_str, league)
    away_team_stats = get_team_stats(db, away_team, season, game_date_str, league)

    # Get prediction
    predictions_collection = league.collections.get('model_predictions', 'nba_model_predictions')
    prediction_doc = db[predictions_collection].find_one({'game_id': game_id})
    last_prediction = None
    if prediction_doc:
        last_prediction = {k: v for k, v in prediction_doc.items() if k != '_id'}

    # Get scores (for completed games) - stored as 'points' in database
    home_score = game.get('homeTeam', {}).get('points')
    away_score = game.get('awayTeam', {}).get('points')

    # Determine game status - if both teams have scores, game is likely completed
    # This is a database-based inference; live status comes from ESPN API polling
    game_status = game.get('status', 'pre')
    game_completed = game.get('completed', False)
    if not game_completed and home_score is not None and away_score is not None:
        game_completed = True
        game_status = 'post'

    return {
        'success': True,
        'game_id': game_id,
        'home_team': home_team_display,  # Display name for UI
        'away_team': away_team_display,  # Display name for UI
        'home_team_id': home_team,  # Internal identifier (abbrev or team_id)
        'away_team_id': away_team,  # Internal identifier (abbrev or team_id)
        'game_date': game_date.strftime('%Y-%m-%d'),
        'gametime': game.get('gametime'),  # Game start time (ISO format)
        'season': season,
        'home_players': home_players,
        'away_players': away_players,
        'home_team_logo': home_team_info['logo'],
        'away_team_logo': away_team_info['logo'],
        'home_team_color': home_team_info['color'],
        'away_team_color': away_team_info['color'],
        'home_team_stats': home_team_stats,
        'away_team_stats': away_team_stats,
        'home_score': home_score,
        'away_score': away_score,
        'status': game_status,
        'completed': game_completed,
        'period': game.get('period'),
        'clock': game.get('clock'),
        'pregame_lines': game.get('pregame_lines'),
        'last_prediction': last_prediction
    }


def get_team_stats(db, team: str, season: str, before_date: str,
                   league: "LeagueConfig") -> Dict:
    """
    Get team record stats (W-L, Home W-L, Away W-L, Last 10 W-L).

    Args:
        db: MongoDB database instance
        team: Team identifier (abbreviation or team_id)
        season: Season string
        before_date: Date string (YYYY-MM-DD) - only count games before this date
        league: LeagueConfig for collection names

    Returns:
        Dictionary with wins, losses, home_wins, home_losses, away_wins, away_losses,
        last10_wins, last10_losses
    """
    games_collection = league.collections.get('games', 'stats_nba')
    team_id_field = league.team_primary_identifier  # 'name' or 'id'
    exclude_game_types = league.exclude_game_types if hasattr(league, 'exclude_game_types') else []

    # Build base query - completed games have homeWon field
    # Match on both primary identifier field AND 'name' for flexibility
    # (CBB games may only have 'name' even though league config says 'id')
    team_match_conditions = [
        {f'homeTeam.{team_id_field}': team},
        {f'awayTeam.{team_id_field}': team}
    ]
    # Also try 'name' field if it's different from primary identifier
    if team_id_field != 'name':
        team_match_conditions.extend([
            {'homeTeam.name': team},
            {'awayTeam.name': team}
        ])

    base_query = {
        'homeWon': {'$exists': True},
        'date': {'$lt': before_date},
        '$or': team_match_conditions
    }
    # Add season filter if provided
    if season:
        base_query['season'] = season
    if exclude_game_types:
        base_query['game_type'] = {'$nin': exclude_game_types}

    # Build home/away match conditions (try both primary field and 'name')
    home_match = {'$or': [{f'homeTeam.{team_id_field}': team}]}
    away_match = {'$or': [{f'awayTeam.{team_id_field}': team}]}
    if team_id_field != 'name':
        home_match['$or'].append({'homeTeam.name': team})
        away_match['$or'].append({'awayTeam.name': team})

    # Use aggregation to count wins/losses efficiently
    pipeline = [
        {'$match': base_query},
        {'$facet': {
            'home_wins': [
                {'$match': {**home_match, 'homeWon': True}},
                {'$count': 'count'}
            ],
            'home_losses': [
                {'$match': {**home_match, 'homeWon': False}},
                {'$count': 'count'}
            ],
            'away_wins': [
                {'$match': {**away_match, 'homeWon': False}},
                {'$count': 'count'}
            ],
            'away_losses': [
                {'$match': {**away_match, 'homeWon': True}},
                {'$count': 'count'}
            ]
        }}
    ]

    result = list(db[games_collection].aggregate(pipeline))

    home_wins = result[0]['home_wins'][0]['count'] if result and result[0]['home_wins'] else 0
    home_losses = result[0]['home_losses'][0]['count'] if result and result[0]['home_losses'] else 0
    away_wins = result[0]['away_wins'][0]['count'] if result and result[0]['away_wins'] else 0
    away_losses = result[0]['away_losses'][0]['count'] if result and result[0]['away_losses'] else 0

    wins = home_wins + away_wins
    losses = home_losses + away_losses

    # Get last 10 games for L10 record
    last10_query = base_query.copy()
    last10_games = list(db[games_collection].find(last10_query).sort('date', -1).limit(10))

    last10_wins = 0
    last10_losses = 0
    for game in last10_games:
        # Try primary field first, then fall back to 'name'
        home_team_id = game.get('homeTeam', {}).get(team_id_field, '') or game.get('homeTeam', {}).get('name', '')
        home_won = game.get('homeWon', False)
        is_home = (home_team_id == team)
        team_won = (is_home and home_won) or (not is_home and not home_won)
        if team_won:
            last10_wins += 1
        else:
            last10_losses += 1

    return {
        'wins': wins,
        'losses': losses,
        'home_wins': home_wins,
        'home_losses': home_losses,
        'away_wins': away_wins,
        'away_losses': away_losses,
        'last10_wins': last10_wins,
        'last10_losses': last10_losses
    }


def get_player_detail(db, player_id: str, team: str, season: str, game_date: date,
                      league: "LeagueConfig") -> Dict:
    """
    Get detailed player stats for the player detail panel.

    Args:
        db: MongoDB database instance
        player_id: Player ID
        team: Team identifier
        season: Season string
        game_date: Date of the game
        league: LeagueConfig for collection names

    Returns:
        Dictionary with player details and stats
    """
    players_collection = league.collections.get('players', 'nba_players')
    player_stats_collection = league.collections.get('player_stats', 'nba_player_stats')

    before_date = game_date.strftime('%Y-%m-%d')
    exclude_game_types = league.exclude_game_types if hasattr(league, 'exclude_game_types') else []

    # Get player info
    player_id_query = [player_id]
    if player_id.isdigit():
        player_id_query.append(int(player_id))

    player_doc = db[players_collection].find_one(
        {'player_id': {'$in': player_id_query}},
        {'player_id': 1, 'player_name': 1, 'headshot': 1, 'pos_name': 1, 'pos_display_name': 1,
         'height': 1, 'weight': 1, 'displayHeight': 1, 'displayWeight': 1}
    ) or {}

    # Calculate stats
    stats = calculate_player_stats(
        db, player_id, team, season, before_date,
        player_stats_collection, exclude_game_types
    )

    # Get height/weight - check both naming conventions
    height = player_doc.get('displayHeight') or player_doc.get('height')
    weight = player_doc.get('displayWeight') or player_doc.get('weight')

    return {
        'success': True,
        'player_id': player_id,
        'player_name': player_doc.get('player_name', ''),
        'headshot': player_doc.get('headshot'),
        'pos_name': player_doc.get('pos_name'),
        'pos_display_name': player_doc.get('pos_display_name'),
        'height': height,
        'weight': weight,
        'team': team,
        'season': season,
        'stats': {
            'ppg': stats.get('ppg', 0.0),
            'rpg': stats.get('rpg', 0.0),
            'apg': stats.get('apg', 0.0),
            'mpg': stats.get('mpg', 0.0),
            'games': stats.get('games', 0),
            'games_started': stats.get('games_started', 0)
        }
    }


def get_player_per(db, player_id: str, team: str, season: str, game_date: date,
                   league: "LeagueConfig") -> Dict:
    """
    Calculate PER for a player. Separate endpoint due to calculation time.

    Args:
        db: MongoDB database instance
        player_id: Player ID
        team: Team identifier
        season: Season string
        game_date: Date of the game
        league: LeagueConfig for collection names

    Returns:
        Dictionary with player PER
    """
    from bball_app.core.stats.per_calculator import PERCalculator

    before_date = game_date.strftime('%Y-%m-%d')

    try:
        # Only preload the specific season, not all seasons
        per_calc = PERCalculator(db, league=league, preload_seasons=[season])
        # Use cross_team=True to include stats from all teams (for traded players)
        per_value = per_calc.get_player_per_before_date(player_id, team, season, before_date, cross_team=True)

        return {
            'success': True,
            'player_id': player_id,
            'per': round(per_value, 1) if per_value is not None else None
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'player_id': player_id,
            'per': None
        }


def get_players_per_batch(db, players: List[Dict], season: str, game_date: date,
                          league: "LeagueConfig") -> Dict[str, float]:
    """
    Calculate PER for multiple players at once. More efficient than individual calls.

    Args:
        db: MongoDB database instance
        players: List of player dicts with 'player_id' and 'team' keys
        season: Season string
        game_date: Date of the game
        league: LeagueConfig for collection names

    Returns:
        Dictionary mapping player_id to PER value (or None if not available)
    """
    from bball_app.core.stats.per_calculator import PERCalculator

    before_date = game_date.strftime('%Y-%m-%d')
    result = {}

    try:
        # Create PERCalculator with only the current season preloaded (not all seasons)
        per_calc = PERCalculator(db, league=league, preload_seasons=[season])

        for player in players:
            player_id = str(player.get('player_id', ''))
            team = player.get('team', '')

            if not player_id:
                continue

            try:
                # Use cross_team=True to include stats from all teams (for traded players)
                per_value = per_calc.get_player_per_before_date(player_id, team, season, before_date, cross_team=True)
                result[player_id] = round(per_value, 1) if per_value is not None else None
            except Exception:
                result[player_id] = None

    except Exception as e:
        print(f"Error calculating batch PER: {e}")

    return result


def search_players_for_roster(db, query: str, season: str,
                              league: "LeagueConfig") -> List[Dict]:
    """
    Search for players eligible to add to a roster.

    Aggregates player_stats to find distinct players matching the query
    who have >1 game with minutes in this season. Enriches results with
    headshot and position from the players collection.

    Args:
        db: MongoDB database instance
        query: Search string (partial player name)
        season: Season string (e.g., '2024-2025')
        league: LeagueConfig for collection names and game type exclusions

    Returns:
        List of player dicts with player_id, player_name, team, headshot,
        pos_name, pos_display_name, games
    """
    from bball_app.core.data.players import PlayerStatsRepository, PlayersRepository

    if len(query) < 2:
        return []

    player_stats_repo = PlayerStatsRepository(db, league=league)
    players_repo = PlayersRepository(db, league=league)

    exclude_game_types = league.exclude_game_types if hasattr(league, 'exclude_game_types') else []

    match_query = {
        'season': season,
        'stats.min': {'$gt': 0},
        'player_name': {'$regex': query, '$options': 'i'}
    }
    if exclude_game_types:
        match_query['game_type'] = {'$nin': exclude_game_types}

    pipeline = [
        {'$match': match_query},
        {'$group': {
            '_id': '$player_id',
            'player_name': {'$first': '$player_name'},
            'team': {'$last': '$team'},
            'games': {'$sum': 1}
        }},
        {'$match': {'games': {'$gt': 1}}},
        {'$sort': {'player_name': 1}},
        {'$limit': 15}
    ]

    results = player_stats_repo.aggregate(pipeline)

    # Enrich with headshot and position from players collection
    player_ids = [str(r['_id']) for r in results]
    players_details = {}
    if player_ids:
        for p in players_repo.find_by_ids(player_ids):
            players_details[str(p.get('player_id', ''))] = p

    response = []
    for r in results:
        pid = str(r['_id'])
        details = players_details.get(pid, {})
        response.append({
            'player_id': pid,
            'player_name': r['player_name'],
            'team': r.get('team', ''),
            'headshot': details.get('headshot'),
            'pos_name': details.get('pos_name'),
            'pos_display_name': details.get('pos_display_name'),
            'games': r['games']
        })

    return response


def add_player_to_team_roster(db, player_id: str, team: str, season: str,
                              game_date: str, league: "LeagueConfig") -> Dict:
    """
    Add a player to a team's roster, removing them from any other team.

    Steps:
    1. Remove player from all other team rosters for this season
    2. Add to target team roster (as non-starter bench player)
    3. Update players collection team field
    4. Build and return full player object for frontend rendering

    Args:
        db: MongoDB database instance
        player_id: Player ID string
        team: Target team identifier
        season: Season string
        game_date: Game date string (YYYY-MM-DD) for stats calculation
        league: LeagueConfig for collection names

    Returns:
        Dict with 'success' bool and 'player' object (or 'error' string)
    """
    from bball_app.core.data.rosters import RostersRepository
    from bball_app.core.data.players import PlayersRepository

    rosters_repo = RostersRepository(db, league=league)
    players_repo = PlayersRepository(db, league=league)

    player_stats_collection = league.collections.get('player_stats', 'nba_player_stats')
    exclude_game_types = league.exclude_game_types if hasattr(league, 'exclude_game_types') else []

    # Remove player from all teams' rosters for this season
    all_rosters = rosters_repo.find_by_season(season)
    for roster_doc in all_rosters:
        roster_team = roster_doc.get('team')
        rosters_repo.remove_player_from_roster(roster_team, season, str(player_id))

    # Add to target team roster
    rosters_repo.add_player_to_roster(team, season, {
        'player_id': str(player_id),
        'starter': False
    })

    # Update players collection team field
    players_repo.upsert_player(player_id, {
        'player_id': player_id,
        'team': team,
        'last_roster_update': datetime.utcnow()
    })

    # Build full player object matching get_team_players return shape
    player_doc = players_repo.find_by_player_id(str(player_id))

    before_date = game_date or date.today().strftime('%Y-%m-%d')
    stats = calculate_player_stats(
        db, str(player_id), team, season, before_date,
        player_stats_collection, exclude_game_types
    )

    player_obj = {
        'player_id': str(player_id),
        'player_name': player_doc.get('player_name', '') if player_doc else '',
        'headshot': player_doc.get('headshot') if player_doc else None,
        'pos_name': player_doc.get('pos_name') if player_doc else None,
        'pos_display_name': player_doc.get('pos_display_name') if player_doc else None,
        'injured': False,
        'injury_status': player_doc.get('injury_status') if player_doc else None,
        'starter': False,
        'stats': stats
    }

    return {'success': True, 'player': player_obj}


def move_player_to_team(db, player_id: str, to_team: str, season: str,
                        league: "LeagueConfig") -> Dict:
    """
    Move a player from their current team to a new team.

    Steps:
    1. Find player's current roster entry (to preserve injured/disabled flags)
    2. Remove from ALL team rosters for the season
    3. Add to target team with preserved injured/disabled flags, starter=False
    4. Update players collection team field

    Args:
        db: MongoDB database instance
        player_id: Player ID string
        to_team: Target team identifier
        season: Season string
        league: LeagueConfig for collection names

    Returns:
        Dict with 'success' bool and 'message' (or 'error' string)
    """
    from bball_app.core.data.rosters import RostersRepository
    from bball_app.core.data.players import PlayersRepository

    rosters_repo = RostersRepository(db, league=league)
    players_repo = PlayersRepository(db, league=league)

    pid_str = str(player_id)

    # Find current roster entry to preserve flags
    source_entry = None
    all_rosters = rosters_repo.find_by_season(season)
    for roster_doc in all_rosters:
        for entry in roster_doc.get('roster', []):
            if str(entry.get('player_id', '')) == pid_str:
                source_entry = entry.copy()
                break
        if source_entry:
            break

    # Remove from ALL team rosters for the season
    for roster_doc in all_rosters:
        roster_team = roster_doc.get('team')
        rosters_repo.remove_player_from_roster(roster_team, season, pid_str)

    # Build new roster entry preserving injured/disabled flags, reset starter
    new_entry = {
        'player_id': pid_str,
        'starter': False,
        'injured': source_entry.get('injured', False) if source_entry else False,
        'disabled': source_entry.get('disabled', False) if source_entry else False,
    }
    # Preserve player_name if it was on the source entry
    if source_entry and source_entry.get('player_name'):
        new_entry['player_name'] = source_entry['player_name']

    rosters_repo.add_player_to_roster(to_team, season, new_entry)

    # Update players collection team field
    players_repo.upsert_player(player_id, {
        'player_id': player_id,
        'team': to_team,
        'last_roster_update': datetime.utcnow()
    })

    return {'success': True, 'message': f'Player {pid_str} moved to {to_team}'}
