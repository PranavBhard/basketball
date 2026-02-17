"""
ID Mapper Service - Builds and manages ID mappings between internal IDs and external source IDs.

This service helps maintain mappings between:
- Internal player IDs <-> ESPN player IDs
- Internal player IDs <-> Basketball-Reference codes
- Internal team IDs <-> Various source codes/slugs

Usage:
    from core.services.id_mapper import IDMapper

    mapper = IDMapper("nba")

    # Build ESPN player ID map from API
    mapper.build_espn_player_map()

    # Look up a player
    espn_id = mapper.get_espn_player_id("internal_player_id")
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from bball_app.core.league_config import load_league_config, LeagueConfig

logger = logging.getLogger(__name__)


def _repo_root() -> str:
    """Return filesystem path to the repo root."""
    nba_app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.dirname(nba_app_dir)


def _nba_app_dir() -> str:
    """Return path to nba_app directory."""
    # core/services/id_mapper.py -> core/services -> core -> nba_app
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _id_maps_dir() -> str:
    """Return path to id_maps directory (nba_app/leagues/id_maps)."""
    return os.path.join(_nba_app_dir(), "leagues", "id_maps")


class IDMapper:
    """
    Service for building and managing ID mappings between internal and external IDs.
    """

    def __init__(self, league: str = "nba"):
        self.league = league
        self.config = load_league_config(league)
        self._maps_dir = _id_maps_dir()
        os.makedirs(self._maps_dir, exist_ok=True)

    def _get_map_path(self, map_name: str) -> str:
        """Get full path for a mapping file."""
        return os.path.join(self._maps_dir, f"{self.league}_{map_name}.json")

    def _load_map(self, map_name: str) -> Dict[str, Any]:
        """Load a mapping file."""
        path = self._get_map_path(map_name)
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def _save_map(self, map_name: str, data: Dict[str, Any]) -> str:
        """Save a mapping file and return the path."""
        path = self._get_map_path(map_name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved mapping to {path}")
        return path

    def build_espn_player_map(self, save: bool = True) -> Dict[str, Dict[str, str]]:
        """
        Build ESPN player ID map from ESPN API.

        Fetches all team rosters and builds a mapping of:
        ESPN player_id -> {name, team, slug, position}

        Args:
            save: Whether to save the map to disk

        Returns:
            Dict mapping ESPN player_id -> player info
        """
        espn_cfg = self.config.espn
        sport_path = espn_cfg.get("sport_path", "nba")

        # Fetch teams
        teams_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/{sport_path}/teams?limit=100"
        teams_resp = requests.get(teams_url, timeout=10)
        teams_resp.raise_for_status()
        teams_data = teams_resp.json()

        teams = teams_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        logger.info(f"Found {len(teams)} teams for {self.league}")

        player_map = {}
        for team in teams:
            team_info = team.get("team", {})
            team_id = team_info.get("id")
            abbrev = team_info.get("abbreviation", "")

            try:
                roster_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/{sport_path}/teams/{team_id}/roster"
                roster_resp = requests.get(roster_url, timeout=10)
                roster_resp.raise_for_status()
                roster_data = roster_resp.json()

                athletes = roster_data.get("athletes", [])
                for athlete in athletes:
                    player_id = str(athlete.get("id", ""))
                    full_name = athlete.get("fullName", "")
                    position = athlete.get("position", {}).get("abbreviation", "")

                    if player_id and full_name:
                        # Generate slug from name
                        slug = re.sub(r"[^a-z0-9\-]", "", full_name.lower().replace(" ", "-"))
                        player_map[player_id] = {
                            "name": full_name,
                            "team": abbrev,
                            "team_id": str(team_id),
                            "slug": slug,
                            "position": position,
                        }

                logger.info(f"  {abbrev}: {len(athletes)} players")
            except Exception as e:
                logger.warning(f"  {abbrev}: Failed to fetch roster - {e}")

        logger.info(f"Total players: {len(player_map)}")

        if save:
            self._save_map("espn_player_ids", player_map)

        return player_map

    def build_espn_team_map(self, save: bool = True) -> Dict[str, Dict[str, str]]:
        """
        Build ESPN team ID map from ESPN API.

        Returns:
            Dict mapping team_id -> {abbreviation, name, slug}
        """
        espn_cfg = self.config.espn
        sport_path = espn_cfg.get("sport_path", "nba")

        teams_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/{sport_path}/teams?limit=500"
        teams_resp = requests.get(teams_url, timeout=10)
        teams_resp.raise_for_status()
        teams_data = teams_resp.json()

        teams = teams_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])

        team_map = {}
        for team in teams:
            team_info = team.get("team", {})
            team_id = str(team_info.get("id", ""))
            abbrev = team_info.get("abbreviation", "")
            name = team_info.get("displayName", "")
            slug = team_info.get("slug", "")

            if team_id:
                team_map[team_id] = {
                    "abbreviation": abbrev,
                    "name": name,
                    "slug": slug,
                }

        logger.info(f"Found {len(team_map)} teams")

        if save:
            self._save_map("espn_team_ids", team_map)

        return team_map

    def get_espn_player_id(self, internal_id: str) -> Optional[str]:
        """Look up ESPN player ID from internal ID."""
        # For now, we assume internal ID = ESPN ID for simplicity
        # In a real implementation, this would look up a reverse mapping
        return internal_id

    def get_espn_player_info(self, espn_player_id: str) -> Optional[Dict[str, str]]:
        """Get player info for an ESPN player ID."""
        player_map = self._load_map("espn_player_ids")
        return player_map.get(espn_player_id)

    def search_player_by_name(self, name: str) -> List[Tuple[str, Dict[str, str]]]:
        """
        Search for players by name (case-insensitive partial match).

        Returns:
            List of (player_id, player_info) tuples
        """
        player_map = self._load_map("espn_player_ids")
        name_lower = name.lower()
        matches = []
        for player_id, info in player_map.items():
            if name_lower in info.get("name", "").lower():
                matches.append((player_id, info))
        return matches


def build_all_maps(league: str = "nba"):
    """Build all ID maps for a league."""
    mapper = IDMapper(league)

    print(f"Building ESPN player map for {league}...")
    player_map = mapper.build_espn_player_map()
    print(f"  Saved {len(player_map)} players")

    print(f"\nBuilding ESPN team map for {league}...")
    team_map = mapper.build_espn_team_map()
    print(f"  Saved {len(team_map)} teams")

    print("\nDone!")


if __name__ == "__main__":
    import sys
    league = sys.argv[1] if len(sys.argv) > 1 else "nba"
    build_all_maps(league)
