"""
ESPN API Client - Wrapper for ESPN NBA API endpoints.

Provides a clean interface to fetch:
- Scoreboard data (games for a date)
- Game summaries (detailed game info)
- Matchup info with venues

This is the data layer abstraction for all ESPN API calls.
"""

import requests
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass
from pytz import timezone

from bball_app.core.league_config import load_league_config

if TYPE_CHECKING:
    from bball_app.core.league_config import LeagueConfig


# Backward-compatible defaults (NBA).
DEFAULT_LEAGUE_ID = "nba"

# Default headers
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Default timeout
DEFAULT_TIMEOUT = 30


@dataclass
class GameInfo:
    """Game information from ESPN API."""
    game_id: str
    home_team: str
    away_team: str
    game_date: str  # YYYY-MM-DD
    gametime: Optional[str] = None
    status: Optional[str] = None  # 'pre', 'in', 'post'
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    venue_guid: Optional[str] = None
    venue_name: Optional[str] = None


@dataclass
class MatchupInfo:
    """Matchup info with venue for predictions."""
    game_id: str
    home_team: str
    away_team: str
    venue_guid: Optional[str] = None


class ESPNClient:
    """
    Client for ESPN NBA API.

    Usage:
        client = ESPNClient()
        games = client.get_games_for_date(date(2024, 1, 15))
        summary = client.get_game_summary('401584701')
    """

    def __init__(
        self,
        league: Union[str, "LeagueConfig", None] = None,
        headers: Dict[str, str] = None,
        timeout: int = DEFAULT_TIMEOUT
    ):
        """
        Initialize ESPN client.

        Args:
            league: LeagueConfig or league_id string. Defaults to NBA.
            headers: Optional custom headers for requests
            timeout: Request timeout in seconds
        """
        if league is None:
            self.league = load_league_config(DEFAULT_LEAGUE_ID)
        elif isinstance(league, str):
            self.league = load_league_config(league)
        else:
            self.league = league

        self.headers = headers or DEFAULT_HEADERS.copy()
        self.timeout = timeout
        self._tz = timezone(self.league.timezone)

    def _url(self, endpoint_key: str, **kwargs) -> str:
        """
        Build a fully formatted URL from league templates.

        League config templates support placeholders:
        - {YYYYMMDD} for dates
        - {game_id} for event id
        """
        base = self.league.espn_endpoint(endpoint_key)
        return base.format(**kwargs)

    # --- Scoreboard Methods ---

    def get_scoreboard_raw(self, game_date: date) -> Optional[Dict]:
        """
        Fetch raw scoreboard data for a date (header API - lightweight, no venue info).

        Args:
            game_date: Date to fetch scoreboard for

        Returns:
            Raw JSON response or None if failed
        """
        date_str = game_date.strftime('%Y%m%d')
        url = self._url("scoreboard_header_template", YYYYMMDD=date_str)

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching scoreboard for {game_date}: {e}")
            return None

    def get_scoreboard_site(self, game_date: date) -> Optional[Dict]:
        """
        Fetch scoreboard data from site API (includes venue info with IDs).

        This API returns more detailed data including:
        - competitions[].venue with id, fullName, address
        - Full team info
        - Odds data

        Args:
            game_date: Date to fetch scoreboard for

        Returns:
            Raw JSON response or None if failed
        """
        date_str = game_date.strftime('%Y%m%d')
        url = self._url("scoreboard_site_template", YYYYMMDD=date_str)

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching site scoreboard for {game_date}: {e}")
            return None

    def get_games_for_date(self, game_date: date) -> List[GameInfo]:
        """
        Get all league games for a specific date.

        Args:
            game_date: Date to fetch games for

        Returns:
            List of GameInfo objects
        """
        scoreboard = self.get_scoreboard_raw(game_date)
        if not scoreboard:
            return []

        games = []
        date_str = game_date.strftime('%Y-%m-%d')
        league_slug = self.league.espn.get("league_slug")

        for sport in scoreboard.get('sports', []):
            if sport.get('slug') != 'basketball':
                continue
            for league in sport.get('leagues', []):
                if league_slug and league.get('slug') != league_slug:
                    continue
                for event in league.get('events', []):
                    game_id = event.get('id')
                    if not game_id:
                        continue

                    # Extract teams
                    competitors = event.get('competitors', [])
                    home_team = None
                    away_team = None
                    home_score = None
                    away_score = None

                    for comp in competitors:
                        abbrev = comp.get('abbreviation', '').upper()
                        score = comp.get('score')
                        if comp.get('homeAway') == 'home':
                            home_team = abbrev
                            home_score = int(score) if score and score.isdigit() else None
                        else:
                            away_team = abbrev
                            away_score = int(score) if score and score.isdigit() else None

                    if not home_team or not away_team:
                        continue

                    # Get game status
                    status = event.get('status', 'pre')
                    gametime = event.get('date')

                    games.append(GameInfo(
                        game_id=str(game_id),
                        home_team=home_team,
                        away_team=away_team,
                        game_date=date_str,
                        gametime=gametime,
                        status=status,
                        home_score=home_score,
                        away_score=away_score
                    ))

        return games

    # --- Teams Methods ---

    def get_teams(self, page: int = 1, limit: int = 500) -> Optional[Dict]:
        """
        Fetch all teams from ESPN teams endpoint.

        Args:
            page: Page number for pagination
            limit: Number of teams per page

        Returns:
            Raw JSON response or None if failed
        """
        url = self._url("teams_template") + f"?page={page}&limit={limit}"

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching teams: {e}")
            return None

    # --- Game Summary Methods ---

    def get_game_summary(self, game_id: str) -> Optional[Dict]:
        """
        Fetch detailed game summary.

        Args:
            game_id: ESPN game ID

        Returns:
            Raw JSON response or None if failed
        """
        url = self._url("game_summary_template", game_id=game_id)

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching game summary for {game_id}: {e}")
            return None

    def get_venue_for_game(self, game_id: str) -> Optional[str]:
        """
        Get venue GUID for a game.

        Args:
            game_id: ESPN game ID

        Returns:
            Venue GUID or None
        """
        summary = self.get_game_summary(game_id)
        if not summary:
            return None

        game_info = summary.get('gameInfo', {})
        venue = game_info.get('venue', {})
        return venue.get('guid')

    # --- Matchup Methods ---

    def get_matchups_for_date(self, game_date: date) -> List[MatchupInfo]:
        """
        Get matchups with venue info for predictions.

        Args:
            game_date: Date to fetch matchups for

        Returns:
            List of MatchupInfo objects with venue GUIDs
        """
        games = self.get_games_for_date(game_date)
        matchups = []

        for game in games:
            # Fetch venue GUID from game summary
            venue_guid = self.get_venue_for_game(game.game_id)

            matchups.append(MatchupInfo(
                game_id=game.game_id,
                home_team=game.home_team,
                away_team=game.away_team,
                venue_guid=venue_guid
            ))

        return matchups

    # --- Utility Methods ---

    @staticmethod
    def parse_espn_utc_to_eastern(date_str: str) -> Optional[datetime]:
        """
        Parse ESPN UTC date string and convert to Eastern time.

        Args:
            date_str: UTC date string from ESPN API (e.g., "2024-01-14T00:30Z")

        Returns:
            datetime in Eastern timezone, or None if parsing fails
        """
        if not date_str:
            return None
        try:
            dt_utc = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            eastern = timezone('America/New_York')
            return dt_utc.astimezone(eastern)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def get_season_from_date(game_date: date, league: "LeagueConfig" = None) -> str:
        """
        Get season string from date using league config's season_cutover_month.

        Args:
            game_date: Date to get season for
            league: Optional LeagueConfig for league-specific cutover month

        Returns:
            Season string (e.g., '2024-2025')
        """
        cutover_month = league.season_rules.get('season_cutover_month', 8) if league else 8
        if game_date.month > cutover_month:
            return f"{game_date.year}-{game_date.year + 1}"
        else:
            return f"{game_date.year - 1}-{game_date.year}"

    def parse_date_range(self, date_str: str) -> List[date]:
        """
        Parse comma-separated date range into list of dates.

        Args:
            date_str: Single date (YYYYMMDD) or range (YYYYMMDD,YYYYMMDD)

        Returns:
            List of date objects
        """
        dates = []
        parts = date_str.split(',')

        if len(parts) == 1:
            d = datetime.strptime(parts[0].strip(), '%Y%m%d').date()
            dates.append(d)
        elif len(parts) == 2:
            start = datetime.strptime(parts[0].strip(), '%Y%m%d').date()
            end = datetime.strptime(parts[1].strip(), '%Y%m%d').date()
            current = start
            while current <= end:
                dates.append(current)
                current += timedelta(days=1)
        else:
            raise ValueError("Invalid date format. Use YYYYMMDD or YYYYMMDD,YYYYMMDD")

        return dates
