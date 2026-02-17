"""
News Service - League-agnostic news and content fetcher.

Fetches team, player, and game-level content from configured news sources
using URL patterns defined in league YAML configs.

Usage:
    from core.services.news_service import NewsService

    # Initialize for a specific league
    news = NewsService("nba")

    # Fetch team-level content
    results = news.fetch_team("LAL", sources=["espn", "rotowire"])

    # Fetch player-level content
    results = news.fetch_player(player_id="1966", sources=["espn"])

    # Fetch game-level content
    results = news.fetch_game(game_id="401705051", sources=["espn"])

    # Get available sources for an entity type
    sources = news.get_sources_for("player")  # Sources with player patterns
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from bball_app.core.league_config import load_league_config, LeagueConfig

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result from fetching content from a single source."""
    source: str
    url: str
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    content_type: Optional[str] = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "url": self.url,
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "content_type": self.content_type,
            "fetched_at": self.fetched_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class NewsResults:
    """Aggregated results from fetching content from multiple sources."""
    entity_type: str  # "team", "player", "game"
    entity_id: str
    league: str
    results: List[FetchResult] = field(default_factory=list)

    @property
    def successful(self) -> List[FetchResult]:
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> List[FetchResult]:
        return [r for r in self.results if not r.success]

    def get_content(self, source: str) -> Optional[str]:
        """Get content from a specific source."""
        for r in self.results:
            if r.source == source and r.success:
                return r.content
        return None

    def all_content(self) -> Dict[str, str]:
        """Get all successful content as {source: content} dict."""
        return {r.source: r.content for r in self.successful if r.content}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "league": self.league,
            "results": [r.to_dict() for r in self.results],
            "successful_count": len(self.successful),
            "failed_count": len(self.failed),
        }


class NewsService:
    """
    League-agnostic service for fetching news and content from configured sources.

    The service reads URL patterns from the league's YAML config and builds
    appropriate URLs for team, player, and game entities. It supports:

    - Multiple sources per entity type
    - Source-specific ID mappings (team codes, player IDs, slugs)
    - Parallel fetching for efficiency
    - Content extraction from HTML

    Example league config structure:

        news_sources:
          espn:
            base_url: https://www.espn.com
            sport_path: nba
            patterns:
              team: /{sport_path}/team/_/name/{team_code}
              player: /{sport_path}/player/_/id/{player_id}
              game: /{sport_path}/game/_/gameId/{game_id}
            mappings:
              team_codes:
                LAL: lal
                BOS: bos
    """

    def __init__(self, league: str = "nba", timeout: int = 10):
        """
        Initialize NewsService for a specific league.

        Args:
            league: League ID (e.g., "nba", "cbb")
            timeout: HTTP request timeout in seconds
        """
        self.league = league
        self.timeout = timeout
        self.config: LeagueConfig = load_league_config(league)
        self._news_sources: Dict[str, Any] = self.config.raw.get("news_sources", {})
        self._id_maps_cache: Dict[str, Dict] = {}

    @property
    def available_sources(self) -> List[str]:
        """List of all configured news sources."""
        return list(self._news_sources.keys())

    def get_source_config(self, source: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific source."""
        return self._news_sources.get(source)

    def get_sources_for(self, entity_type: str) -> List[str]:
        """
        Get list of sources that have patterns for a given entity type.

        Args:
            entity_type: "team", "player", or "game"

        Returns:
            List of source names that support this entity type
        """
        sources = []
        for name, cfg in self._news_sources.items():
            patterns = cfg.get("patterns", {})
            # Check if any pattern key starts with entity_type
            if any(k.startswith(entity_type) for k in patterns.keys()):
                sources.append(name)
        return sources

    def get_patterns_for_source(self, source: str, entity_type: str) -> Dict[str, str]:
        """
        Get all patterns for a source and entity type.

        Args:
            source: Source name (e.g., "espn")
            entity_type: "team", "player", or "game"

        Returns:
            Dict of pattern_name -> pattern_template
        """
        cfg = self._news_sources.get(source, {})
        patterns = cfg.get("patterns", {})
        return {k: v for k, v in patterns.items() if k.startswith(entity_type)}

    def _load_id_map(self, map_file: str) -> Dict[str, str]:
        """Load an ID mapping file (JSON) with caching."""
        if map_file in self._id_maps_cache:
            return self._id_maps_cache[map_file]

        # Try multiple path resolutions
        from bball_app.core.league_config import _repo_root

        # nba_app directory
        nba_app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Try paths in order: nba_app-relative, repo-root-relative, absolute
        candidate_paths = [
            os.path.join(nba_app_dir, map_file),  # nba_app/leagues/id_maps/...
            os.path.join(_repo_root(), map_file),  # repo_root/leagues/id_maps/...
            map_file,  # absolute path
        ]

        full_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                full_path = path
                break

        if not full_path:
            logger.debug(f"ID map file not found: {map_file} (tried {len(candidate_paths)} locations)")
            return {}

        try:
            with open(full_path, "r") as f:
                data = json.load(f)
            self._id_maps_cache[map_file] = data
            return data
        except Exception as e:
            logger.error(f"Failed to load ID map {full_path}: {e}")
            return {}

    def _get_team_code(self, source: str, team_id: str) -> Optional[str]:
        """
        Get source-specific team code for an internal team ID.

        Args:
            source: Source name
            team_id: Internal team ID/abbreviation

        Returns:
            Source-specific team code or None if not found
        """
        cfg = self._news_sources.get(source, {})
        mappings = cfg.get("mappings", {})

        # Try inline mappings first
        team_codes = mappings.get("team_codes", {})
        if team_id in team_codes:
            return team_codes[team_id]

        # Try team_slugs
        team_slugs = mappings.get("team_slugs", {})
        if team_id in team_slugs:
            return team_slugs[team_id]

        # Try loading from file
        codes_file = mappings.get("team_codes_file") or mappings.get("team_ids_file")
        if codes_file:
            file_map = self._load_id_map(codes_file)
            if team_id in file_map:
                return file_map[team_id]

        # Fallback: return as-is (lowercase for some sources)
        return team_id.lower() if team_id else None

    def _get_team_slug(self, source: str, team_id: str) -> Optional[str]:
        """Get source-specific team slug for an internal team ID."""
        cfg = self._news_sources.get(source, {})
        mappings = cfg.get("mappings", {})

        team_slugs = mappings.get("team_slugs", {})
        if team_id in team_slugs:
            return team_slugs[team_id]

        slugs_file = mappings.get("team_slugs_file")
        if slugs_file:
            file_map = self._load_id_map(slugs_file)
            if team_id in file_map:
                return file_map[team_id]

        return None

    def _get_player_id(self, source: str, player_id: str) -> Optional[str]:
        """Get source-specific player ID for an internal player ID."""
        cfg = self._news_sources.get(source, {})
        mappings = cfg.get("mappings", {})

        # Try loading from file
        ids_file = mappings.get("player_ids_file") or mappings.get("player_codes_file")
        if ids_file:
            file_map = self._load_id_map(ids_file)
            if player_id in file_map:
                value = file_map[player_id]
                # If value is a dict (player info), the key IS the source ID
                # If value is a string, it's the translated source ID
                if isinstance(value, dict):
                    return player_id  # Key is already the source ID
                return value

        # Fallback: return as-is
        return player_id

    def _get_player_info(self, source: str, player_id: str) -> Optional[Dict[str, Any]]:
        """Get player info (name, slug, team, etc.) from ID map."""
        cfg = self._news_sources.get(source, {})
        mappings = cfg.get("mappings", {})

        ids_file = mappings.get("player_ids_file") or mappings.get("player_codes_file")
        if ids_file:
            file_map = self._load_id_map(ids_file)
            if player_id in file_map:
                value = file_map[player_id]
                if isinstance(value, dict):
                    return value
        return None

    def _get_player_slug(self, source: str, player_id: str) -> Optional[str]:
        """Get player slug from ID map."""
        info = self._get_player_info(source, player_id)
        if info:
            return info.get("slug")
        return None

    def build_url(
        self,
        source: str,
        pattern_key: str,
        **kwargs
    ) -> Optional[str]:
        """
        Build a URL from a source pattern and parameters.

        Args:
            source: Source name (e.g., "espn")
            pattern_key: Pattern key (e.g., "team", "player_stats", "game_boxscore")
            **kwargs: Template variables (team_code, player_id, game_id, etc.)

        Returns:
            Fully constructed URL or None if pattern not found
        """
        cfg = self._news_sources.get(source)
        if not cfg:
            logger.warning(f"Source not found: {source}")
            return None

        patterns = cfg.get("patterns", {})
        pattern = patterns.get(pattern_key)
        if not pattern:
            logger.debug(f"Pattern '{pattern_key}' not found in source '{source}'")
            return None

        base_url = cfg.get("base_url", "")
        sport_path = cfg.get("sport_path", "")

        # Build substitution dict
        subs = {
            "sport_path": sport_path,
            **kwargs
        }

        # Handle partial substitution (keep unmatched placeholders)
        class SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"

        try:
            path = pattern.format_map(SafeDict(subs))
            url = f"{base_url}{path}"
            return url
        except Exception as e:
            logger.error(f"Failed to build URL for {source}/{pattern_key}: {e}")
            return None

    def build_team_url(
        self,
        source: str,
        team_id: str,
        pattern_key: str = "team"
    ) -> Optional[str]:
        """
        Build a team URL for a source.

        Args:
            source: Source name
            team_id: Internal team ID/abbreviation
            pattern_key: Pattern key (default "team", could be "team_news", etc.)

        Returns:
            Constructed URL or None
        """
        team_code = self._get_team_code(source, team_id)
        team_slug = self._get_team_slug(source, team_id)

        return self.build_url(
            source,
            pattern_key,
            team_code=team_code,
            team_slug=team_slug,
            team_id=team_id,  # Some sources use numeric IDs
        )

    def build_player_url(
        self,
        source: str,
        player_id: str,
        pattern_key: str = "player",
        player_slug: Optional[str] = None
    ) -> Optional[str]:
        """
        Build a player URL for a source.

        Args:
            source: Source name
            player_id: Internal player ID
            pattern_key: Pattern key (default "player")
            player_slug: Optional player slug for URL

        Returns:
            Constructed URL or None
        """
        source_player_id = self._get_player_id(source, player_id)

        # Try to get slug from player info map if not provided
        if not player_slug:
            player_slug = self._get_player_slug(source, player_id)
        if not player_slug:
            player_slug = source_player_id  # Fallback to ID

        return self.build_url(
            source,
            pattern_key,
            player_id=source_player_id,
            player_slug=player_slug,
        )

    def build_game_url(
        self,
        source: str,
        game_id: str,
        pattern_key: str = "game",
        **kwargs
    ) -> Optional[str]:
        """
        Build a game URL for a source.

        Args:
            source: Source name
            game_id: Internal game ID
            pattern_key: Pattern key (default "game", could be "game_boxscore")
            **kwargs: Additional params (home_code, away_code, game_date, etc.)

        Returns:
            Constructed URL or None
        """
        return self.build_url(
            source,
            pattern_key,
            game_id=game_id,
            **kwargs
        )

    def _fetch_url(self, source: str, url: str, pattern_key: str) -> FetchResult:
        """
        Fetch content from a URL.

        Args:
            source: Source name for logging/result
            url: URL to fetch
            pattern_key: Pattern that was used (for metadata)

        Returns:
            FetchResult with content or error
        """
        cfg = self._news_sources.get(source, {})
        js_required = cfg.get("js_required", False)

        if js_required:
            return FetchResult(
                source=source,
                url=url,
                success=False,
                error="Source requires JavaScript rendering (headless browser needed)",
                metadata={"pattern": pattern_key, "js_required": True}
            )

        try:
            from bball_app.core.services.webpage_parser import WebpageParser
            content = WebpageParser.extract_from_url(url, timeout=self.timeout)

            return FetchResult(
                source=source,
                url=url,
                success=True,
                content=content,
                content_type="text",
                metadata={"pattern": pattern_key}
            )
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return FetchResult(
                source=source,
                url=url,
                success=False,
                error=str(e),
                metadata={"pattern": pattern_key}
            )

    def fetch_team(
        self,
        team_id: str,
        sources: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        parallel: bool = True
    ) -> NewsResults:
        """
        Fetch team-level content from configured sources.

        Args:
            team_id: Internal team ID/abbreviation (e.g., "LAL")
            sources: List of sources to use (None = all with team patterns)
            patterns: List of pattern keys (None = ["team"])
            parallel: Whether to fetch in parallel

        Returns:
            NewsResults with content from each source
        """
        if sources is None:
            sources = self.get_sources_for("team")

        if patterns is None:
            patterns = ["team"]

        results = NewsResults(
            entity_type="team",
            entity_id=team_id,
            league=self.league
        )

        # Build all URLs first
        fetch_jobs = []
        for source in sources:
            for pattern_key in patterns:
                url = self.build_team_url(source, team_id, pattern_key)
                if url:
                    fetch_jobs.append((source, url, pattern_key))

        # Fetch content
        if parallel and len(fetch_jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(len(fetch_jobs), 5)) as executor:
                futures = {
                    executor.submit(self._fetch_url, src, url, pat): (src, url, pat)
                    for src, url, pat in fetch_jobs
                }
                for future in as_completed(futures):
                    results.results.append(future.result())
        else:
            for source, url, pattern_key in fetch_jobs:
                results.results.append(self._fetch_url(source, url, pattern_key))

        return results

    def fetch_player(
        self,
        player_id: str,
        sources: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        player_slug: Optional[str] = None,
        parallel: bool = True
    ) -> NewsResults:
        """
        Fetch player-level content from configured sources.

        Args:
            player_id: Internal player ID
            sources: List of sources to use (None = all with player patterns)
            patterns: List of pattern keys (None = ["player"])
            player_slug: Optional slug for URL building
            parallel: Whether to fetch in parallel

        Returns:
            NewsResults with content from each source
        """
        if sources is None:
            sources = self.get_sources_for("player")

        if patterns is None:
            patterns = ["player"]

        results = NewsResults(
            entity_type="player",
            entity_id=player_id,
            league=self.league
        )

        fetch_jobs = []
        for source in sources:
            for pattern_key in patterns:
                url = self.build_player_url(source, player_id, pattern_key, player_slug)
                if url:
                    fetch_jobs.append((source, url, pattern_key))

        if parallel and len(fetch_jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(len(fetch_jobs), 5)) as executor:
                futures = {
                    executor.submit(self._fetch_url, src, url, pat): (src, url, pat)
                    for src, url, pat in fetch_jobs
                }
                for future in as_completed(futures):
                    results.results.append(future.result())
        else:
            for source, url, pattern_key in fetch_jobs:
                results.results.append(self._fetch_url(source, url, pattern_key))

        return results

    def fetch_game(
        self,
        game_id: str,
        sources: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        parallel: bool = True,
        **kwargs
    ) -> NewsResults:
        """
        Fetch game-level content from configured sources.

        Args:
            game_id: Internal game ID
            sources: List of sources to use (None = all with game patterns)
            patterns: List of pattern keys (None = ["game"])
            parallel: Whether to fetch in parallel
            **kwargs: Additional params for URL building (home_code, game_date, etc.)

        Returns:
            NewsResults with content from each source
        """
        if sources is None:
            sources = self.get_sources_for("game")

        if patterns is None:
            patterns = ["game"]

        results = NewsResults(
            entity_type="game",
            entity_id=game_id,
            league=self.league
        )

        fetch_jobs = []
        for source in sources:
            for pattern_key in patterns:
                url = self.build_game_url(source, game_id, pattern_key, **kwargs)
                if url:
                    fetch_jobs.append((source, url, pattern_key))

        if parallel and len(fetch_jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(len(fetch_jobs), 5)) as executor:
                futures = {
                    executor.submit(self._fetch_url, src, url, pat): (src, url, pat)
                    for src, url, pat in fetch_jobs
                }
                for future in as_completed(futures):
                    results.results.append(future.result())
        else:
            for source, url, pattern_key in fetch_jobs:
                results.results.append(self._fetch_url(source, url, pattern_key))

        return results

    def get_team_urls(self, team_id: str, sources: Optional[List[str]] = None) -> Dict[str, Dict[str, str]]:
        """
        Get all team URLs for a team without fetching content.

        Args:
            team_id: Internal team ID
            sources: Sources to include (None = all)

        Returns:
            Dict of {source: {pattern: url}}
        """
        if sources is None:
            sources = self.get_sources_for("team")

        urls = {}
        for source in sources:
            patterns = self.get_patterns_for_source(source, "team")
            source_urls = {}
            for pattern_key in patterns:
                url = self.build_team_url(source, team_id, pattern_key)
                if url:
                    source_urls[pattern_key] = url
            if source_urls:
                urls[source] = source_urls

        return urls

    def get_player_urls(self, player_id: str, sources: Optional[List[str]] = None) -> Dict[str, Dict[str, str]]:
        """
        Get all player URLs for a player without fetching content.

        Args:
            player_id: Internal player ID
            sources: Sources to include (None = all)

        Returns:
            Dict of {source: {pattern: url}}
        """
        if sources is None:
            sources = self.get_sources_for("player")

        urls = {}
        for source in sources:
            patterns = self.get_patterns_for_source(source, "player")
            source_urls = {}
            for pattern_key in patterns:
                url = self.build_player_url(source, player_id, pattern_key)
                if url:
                    source_urls[pattern_key] = url
            if source_urls:
                urls[source] = source_urls

        return urls

    def get_game_urls(self, game_id: str, sources: Optional[List[str]] = None, **kwargs) -> Dict[str, Dict[str, str]]:
        """
        Get all game URLs for a game without fetching content.

        Args:
            game_id: Internal game ID
            sources: Sources to include (None = all)
            **kwargs: Additional URL params

        Returns:
            Dict of {source: {pattern: url}}
        """
        if sources is None:
            sources = self.get_sources_for("game")

        urls = {}
        for source in sources:
            patterns = self.get_patterns_for_source(source, "game")
            source_urls = {}
            for pattern_key in patterns:
                url = self.build_game_url(source, game_id, pattern_key, **kwargs)
                if url:
                    source_urls[pattern_key] = url
            if source_urls:
                urls[source] = source_urls

        return urls
