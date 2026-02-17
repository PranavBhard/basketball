"""
News / web search tools for matchup agents.

These tools are thin wrappers that:
- Use core NewsService to fetch content from configured sources
- Optionally supplement with SERP API for broader web search
- Return normalized results: [{source, content, metadata}, ...]

Primary sources (via NewsService):
- Team news: HoopsHype (rumors, salaries), web search fallback
- Player news: HoopsHype (rumors), web search fallback
- Game news: ESPN box scores, recaps (has actual written content)

Note: ESPN team/player pages are excluded by default because they
contain stats/metadata rather than news articles or written content.
Use `sources=["espn"]` explicitly if you need stats pages.

Fallback (via SERP API):
- web_search for general queries
- _news_search for news-specific searches

Caching:
- in-memory TTL cache (1 hour) keyed by league_id + entity/query
- force_refresh bypasses cache
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from bball_app.config import config
from bball_app.core.mongo import Mongo
from bball_app.core.services.webpage_parser import WebpageParser
from bball_app.core.services.news_service import NewsService
from bball_app.core.league_config import load_league_config

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


_CACHE: Dict[str, _CacheEntry] = {}
_DEFAULT_TTL_S = 60 * 60  # 1 hour
_MIN_EXTRACT_CHARS = 400  # heuristic: avoid thin/snippet-only pages


def _cache_get(key: str) -> Optional[Any]:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    if time.time() > entry.expires_at:
        _CACHE.pop(key, None)
        return None
    return entry.value


def _cache_set(key: str, value: Any, ttl_s: int = _DEFAULT_TTL_S) -> None:
    _CACHE[key] = _CacheEntry(value=value, expires_at=time.time() + ttl_s)


def _normalize_query(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


_HOMEPAGE_LIKE_PATH_RE = re.compile(
    r"/(news|rumors|schedule|standings|roster|injuries|stats|players?|team|teams)(/)?$",
    re.I,
)


def _is_article_like_url(url: str, title: str = "", snippet: str = "") -> bool:
    """
    Heuristic filter to prefer direct articles over hub pages.
    Designed to be conservative: reject obvious hubs, keep plausible articles.
    """
    try:
        u = urlparse(url or "")
        host = (u.netloc or "").lower()
        path = (u.path or "").strip()
        if not host or not path:
            return False

        # Reject obvious hubs/feeds
        if path in {"/", ""}:
            return False
        if _HOMEPAGE_LIKE_PATH_RE.search(path):
            return False

        # Prefer explicit article indicators
        if re.search(r"/\d{4}/\d{2}/\d{2}/", path):
            return True
        if re.search(r"/\d{4}-\d{2}-\d{2}", path):
            return True
        if path.endswith(".html") or path.endswith(".htm"):
            return True
        if re.search(r"/(story|article|post|posts)/", path, re.I):
            return True

        # If URL path is reasonably deep, accept
        depth = len([p for p in path.split("/") if p])
        if depth >= 3:
            return True

        # Low-depth paths are often hubs; keep only if title/snippet looks like a single item
        if len((title or "").strip()) >= 25 and len((snippet or "").strip()) >= 60:
            return True
    except Exception:
        return False
    return False


def _fetch_and_extract(url: str, *, timeout: int = 10) -> str:
    """
    Fetch HTML in tool layer and parse with core WebpageParser.
    """
    resp2 = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
    )
    resp2.raise_for_status()
    if getattr(resp2, "encoding", None):
        html = resp2.text
    else:
        html = resp2.content.decode("utf-8", errors="replace")
    return WebpageParser.extract_from_html(html)


def _normalize_terms(terms: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for t in terms or []:
        s = (t or "").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _content_is_relevant(content: str, *, required_terms: List[str]) -> bool:
    """
    Best-effort relevance gate:
    - require extracted content to be non-trivial length
    - require at least one required term match (case-insensitive substring)
    """
    if not content:
        return False
    if len(content.strip()) < _MIN_EXTRACT_CHARS:
        return False
    terms = _normalize_terms(required_terms)
    if not terms:
        return True
    c = content.lower()
    return any((t.lower() in c) for t in terms)


def _serpapi_search(
    query: str,
    *,
    force_refresh: bool,
    num_results: int,
    league_id: str,
    tbm: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Raw SERP API search. When tbm == 'nws', requests News vertical.

    Returns a list of result dicts with keys: {source, title, link, snippet, position}.
    """
    league_id = (league_id or "nba").lower()
    query_n = _normalize_query(query)
    mode = f"tbm={tbm}" if tbm else "tbm=web"
    cache_key = f"{league_id}:serp:{mode}:{query_n}:{num_results}"

    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    api_key = config.get("SERP_API_KEY") or config.get("serp_api_key") or config.get("serpapi_key")
    if not api_key:
        return [
            {
                "source": "serpapi",
                "title": "",
                "link": "",
                "snippet": "",
                "position": None,
                "metadata": {"error": "Missing SERP_API_KEY in config.py", "query": query},
            }
        ]

    params = {"q": query, "hl": "en", "gl": "us", "api_key": api_key}
    if tbm:
        params["tbm"] = tbm

    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        data = resp.json() if resp is not None else {}
    except Exception as e:
        return [{"source": "serpapi", "title": "", "link": "", "snippet": "", "position": None, "metadata": {"error": str(e), "query": query}}]

    results: List[Dict[str, Any]] = []

    # Prefer news_results when available
    news_items = data.get("news_results") or []
    if isinstance(news_items, list) and news_items:
        for item in news_items[: max(0, int(num_results))]:
            results.append(
                {
                    "source": item.get("source") or item.get("publisher") or "web",
                    "title": item.get("title") or "",
                    "link": item.get("link") or "",
                    "snippet": item.get("snippet") or "",
                    "position": item.get("position"),
                }
            )

    if not results:
        for item in (data.get("organic_results") or [])[: max(0, int(num_results))]:
            results.append(
                {
                    "source": item.get("source") or item.get("displayed_link") or "web",
                    "title": item.get("title") or "",
                    "link": item.get("link") or "",
                    "snippet": item.get("snippet") or "",
                    "position": item.get("position"),
                }
            )

    _cache_set(cache_key, results)
    return results


def _get_coll(league, key: str, fallback: str) -> str:
    if league is not None and getattr(league, "collections", None):
        return league.collections.get(key) or fallback
    return fallback


def web_search(
    query: str,
    *,
    force_refresh: bool = False,
    num_results: int = 5,
    league_id: str = "nba",
) -> List[Dict[str, Any]]:
    """
    Search via SERP API and return normalized extracted text results.

    Returns list of dicts: {source, content, metadata}
    """
    league_id = (league_id or "nba").lower()
    query_n = _normalize_query(query)
    cache_key = f"{league_id}:web_search:{query_n}:{num_results}"

    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    raw = _serpapi_search(query, force_refresh=force_refresh, num_results=num_results, league_id=league_id, tbm=None)
    out: List[Dict[str, Any]] = []
    for item in raw[: max(0, int(num_results))]:
        link = item.get("link") or ""
        source = item.get("source") or "web"
        snippet = item.get("snippet") or ""
        title = item.get("title") or ""

        content = ""
        if link:
            try:
                content = _fetch_and_extract(link, timeout=10)
            except Exception:
                content = ""

        out.append(
            {
                "source": source,
                "content": content or snippet,
                "metadata": {
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "position": item.get("position"),
                },
            }
        )

    _cache_set(cache_key, out)
    return out


def _news_search(
    query: str,
    *,
    force_refresh: bool,
    num_results: int = 5,
    league_id: str,
    required_terms: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    SERP API News vertical search that prefers direct article links.
    """
    required_terms = required_terms or []

    # Fetch more candidates, then filter down.
    # We will *skip* results whose content fails to load or doesn't pass relevance checks,
    # continuing until we have num_results good items (or candidates are exhausted).
    raw = _serpapi_search(
        query,
        force_refresh=force_refresh,
        num_results=max(20, num_results * 8),
        league_id=league_id,
        tbm="nws",
    )

    # Filter for article-like URLs and take top N
    candidates = []
    for item in raw:
        link = item.get("link") or ""
        title = item.get("title") or ""
        snippet = item.get("snippet") or ""
        if link and _is_article_like_url(link, title=title, snippet=snippet):
            candidates.append(item)
    if not candidates:
        # fallback to raw list if filter is too strict
        candidates = raw

    out: List[Dict[str, Any]] = []
    # Avoid returning multiple links from the same domain unless needed.
    used_hosts = set()
    max_attempts = max(25, int(num_results) * 10)
    attempts = 0

    for item in candidates:
        if len(out) >= max(0, int(num_results)):
            break
        attempts += 1
        if attempts > max_attempts:
            break

        link = item.get("link") or ""
        source = item.get("source") or "web"
        snippet = item.get("snippet") or ""
        title = item.get("title") or ""

        if not link:
            continue

        # Prefer diversity across sources
        try:
            host = (urlparse(link).netloc or "").lower()
        except Exception:
            host = ""
        if host and host in used_hosts and len(out) < max(0, int(num_results)) - 1:
            # keep one "slot" flexible in case we can't fill all 5 uniquely
            continue

        # Must successfully load + extract, and content must match relevance terms (if provided).
        try:
            content = _fetch_and_extract(link, timeout=10)
        except Exception:
            continue
        if not _content_is_relevant(content, required_terms=required_terms):
            continue

        if host:
            used_hosts.add(host)

        out.append(
            {
                "source": source,
                "content": content,
                "metadata": {
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "position": item.get("position"),
                },
            }
        )
    # IMPORTANT: never return an empty list unless SERP itself failed.
    # If the quality gates are too strict (content failed to extract, required_terms mismatch, etc),
    # fall back to general web search results (which include snippet fallback).
    if not out:
        return web_search(query, force_refresh=force_refresh, num_results=num_results, league_id=league_id)
    return out


def get_game_news(
    game_id: str,
    *,
    force_refresh: bool = False,
    league=None,
    league_id: str = "nba",
    db=None,
    include_web_search: bool = False,
    num_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch game-level news and content from configured sources.

    Primary sources (via NewsService):
    - ESPN game page, boxscore, recap
    - Other configured game sources

    Fallback (if include_web_search=True or not enough results):
    - SERP API news search

    Args:
        game_id: ESPN game ID
        force_refresh: Bypass cache
        league: Optional LeagueConfig object
        league_id: League identifier (nba, cbb, etc.)
        db: Optional MongoDB database
        include_web_search: Also search web for additional context
        num_results: Target number of results

    Returns:
        List of {source, content, metadata} dicts
    """
    league_id = (league_id or "nba").lower()
    if db is None:
        db = Mongo().db
    cache_key = f"{league_id}:game_news:{game_id}"

    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    results: List[Dict[str, Any]] = []

    # 1. Fetch from configured news sources via NewsService
    try:
        news_svc = NewsService(league_id)
        # Fetch game page, boxscore, and recap
        fetch_results = news_svc.fetch_game(
            game_id,
            sources=["espn"],  # ESPN is most reliable for game data
            patterns=["game", "game_boxscore", "game_recap"],
            parallel=True,
        )

        for r in fetch_results.successful:
            if r.content and len(r.content.strip()) >= _MIN_EXTRACT_CHARS:
                results.append({
                    "source": r.source,
                    "content": r.content,
                    "metadata": {
                        "title": f"Game {game_id} - {r.metadata.get('pattern', 'page')}",
                        "link": r.url,
                        "pattern": r.metadata.get("pattern"),
                        "fetched_via": "news_service",
                    },
                })
    except Exception as e:
        logger.warning(f"NewsService fetch failed for game {game_id}: {e}")

    # 2. If we need more results or web search requested, use SERP API
    if include_web_search or len(results) < num_results:
        games_coll = _get_coll(league, "games", "stats_nba")
        teams_coll = _get_coll(league, "teams", "teams_nba")

        game = db[games_coll].find_one({"game_id": game_id}) or {}
        date_str = str(game.get("date") or "")[:10]
        away_abbrev = (game.get("awayTeam") or {}).get("name") or ""
        home_abbrev = (game.get("homeTeam") or {}).get("name") or ""

        away_doc = db[teams_coll].find_one({"abbreviation": away_abbrev}) or {}
        home_doc = db[teams_coll].find_one({"abbreviation": home_abbrev}) or {}
        away_full = away_doc.get("displayName") or away_abbrev
        home_full = home_doc.get("displayName") or home_abbrev

        # Get sport context from league config for more accurate searches
        # This is especially important for college sports where schools have multiple sports programs
        sport_context = ""
        try:
            league_cfg = load_league_config(league_id)
            sport = league_cfg.raw.get("meta", {}).get("sport", "")
            display_name = league_cfg.raw.get("meta", {}).get("display_name", "")
            # For college sports, add sport to disambiguate from football/other sports
            if "college" in display_name.lower() or league_id in ("cbb", "cfb", "cws"):
                sport_context = f" {sport}" if sport else " basketball"
            elif league_id not in ("nba", "nfl", "mlb", "nhl"):
                # For other non-major leagues, also include sport context
                sport_context = f" {sport}" if sport else ""
        except Exception:
            pass

        if away_full or home_full:
            q = f"{away_full} at {home_full}{sport_context} {date_str} preview odds prediction".strip()
            web_results = _news_search(
                q,
                force_refresh=force_refresh,
                num_results=max(1, num_results - len(results)),
                league_id=league_id,
                required_terms=[away_full, home_full, away_abbrev, home_abbrev],
            )
            # Add web results, avoiding duplicates
            existing_links = {r.get("metadata", {}).get("link") for r in results}
            for wr in web_results:
                link = wr.get("metadata", {}).get("link")
                if link not in existing_links:
                    results.append(wr)
                    existing_links.add(link)

    _cache_set(cache_key, results)
    return results


def get_team_news(
    team_id: str,
    *,
    force_refresh: bool = False,
    league=None,
    league_id: str = "nba",
    db=None,
    include_web_search: bool = False,
    num_results: int = 5,
    sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch team-level news and content from configured sources.

    Primary sources (via NewsService):
    - HoopsHype rumors and salaries (has actual written content)

    Note: ESPN team pages are excluded by default because they contain
    stats/metadata rather than news articles. Use sources=["espn"] explicitly
    if you need stats pages.

    Fallback (if include_web_search=True or not enough results):
    - SERP API news search

    Args:
        team_id: Team abbreviation (LAL, BOS) or ESPN team_id
        force_refresh: Bypass cache
        league: Optional LeagueConfig object
        league_id: League identifier (nba, cbb, etc.)
        db: Optional MongoDB database
        include_web_search: Also search web for additional context
        num_results: Target number of results
        sources: Specific sources to use (default: hoopshype)

    Returns:
        List of {source, content, metadata} dicts
    """
    league_id = (league_id or "nba").lower()
    if db is None:
        db = Mongo().db
    cache_key = f"{league_id}:team_news:{team_id}"

    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    results: List[Dict[str, Any]] = []

    # Resolve team info from DB
    teams_coll = _get_coll(league, "teams", "teams_nba")
    team = db[teams_coll].find_one({"team_id": team_id}) or db[teams_coll].find_one({"id": team_id}) or {}

    # Also try by abbreviation if team_id lookup failed
    if not team:
        team = db[teams_coll].find_one({"abbreviation": team_id}) or {}

    full_name = team.get("displayName") or team.get("name") or ""
    abbrev = team.get("abbreviation") or team.get("abbr") or team_id

    # 1. Fetch from configured news sources via NewsService
    try:
        news_svc = NewsService(league_id)

        # Default sources: HoopsHype (rumors/salaries with actual written content)
        # ESPN excluded by default - team pages are stats, not news articles
        if sources is None:
            sources = ["hoopshype"]

        # Fetch team page
        fetch_results = news_svc.fetch_team(
            abbrev,  # Use abbreviation for team lookups
            sources=sources,
            patterns=["team"],
            parallel=True,
        )

        for r in fetch_results.successful:
            if r.content and len(r.content.strip()) >= _MIN_EXTRACT_CHARS:
                results.append({
                    "source": r.source,
                    "content": r.content,
                    "metadata": {
                        "title": f"{full_name or abbrev} - {r.metadata.get('pattern', 'page')}",
                        "link": r.url,
                        "pattern": r.metadata.get("pattern"),
                        "fetched_via": "news_service",
                    },
                })
    except Exception as e:
        logger.warning(f"NewsService fetch failed for team {team_id}: {e}")

    # 2. If we need more results or web search requested, use SERP API
    if include_web_search or len(results) < num_results:
        # Get sport context from league config for more accurate searches
        # This is especially important for college sports where schools have multiple sports programs
        sport_context = ""
        try:
            league_cfg = load_league_config(league_id)
            sport = league_cfg.raw.get("meta", {}).get("sport", "")
            display_name = league_cfg.raw.get("meta", {}).get("display_name", "")
            # For college sports, add sport to disambiguate from football/other sports
            if "college" in display_name.lower() or league_id in ("cbb", "cfb", "cws"):
                sport_context = f" {sport}" if sport else " basketball"
            elif league_id not in ("nba", "nfl", "mlb", "nhl"):
                # For other non-major leagues, also include sport context
                sport_context = f" {sport}" if sport else ""
        except Exception:
            pass

        if full_name:
            q = f"{full_name}{sport_context} news and latest".strip()
        else:
            q = f"{team_id}{sport_context} news and latest".strip()

        web_results = _news_search(
            q,
            force_refresh=force_refresh,
            num_results=max(1, num_results - len(results)),
            league_id=league_id,
            required_terms=[full_name, abbrev] if full_name else [],
        )
        # Add web results, avoiding duplicates
        existing_links = {r.get("metadata", {}).get("link") for r in results}
        for wr in web_results:
            link = wr.get("metadata", {}).get("link")
            if link not in existing_links:
                results.append(wr)
                existing_links.add(link)

    _cache_set(cache_key, results)
    return results


def get_player_news(
    player_id: str,
    *,
    force_refresh: bool = False,
    league=None,
    league_id: str = "nba",
    db=None,
    include_web_search: bool = False,
    num_results: int = 5,
    sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch player-level news and content from configured sources.

    Primary sources (via NewsService):
    - HoopsHype player rumors and salary info

    Note: ESPN player pages are excluded by default because they contain
    stats/metadata rather than news articles. Use sources=["espn"] explicitly
    if you need stats pages.

    Fallback (if include_web_search=True or not enough results):
    - SERP API news search

    Args:
        player_id: ESPN player ID
        force_refresh: Bypass cache
        league: Optional LeagueConfig object
        league_id: League identifier (nba, cbb, etc.)
        db: Optional MongoDB database
        include_web_search: Also search web for additional context
        num_results: Target number of results
        sources: Specific sources to use (default: hoopshype)

    Returns:
        List of {source, content, metadata} dicts
    """
    league_id = (league_id or "nba").lower()
    if db is None:
        db = Mongo().db
    cache_key = f"{league_id}:player_news:{player_id}"

    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    results: List[Dict[str, Any]] = []

    # Resolve player info from DB
    players_coll = _get_coll(league, "players", "players_nba")
    player = db[players_coll].find_one({"player_id": str(player_id)}) or {}
    name = player.get("player_name") or player.get("name") or ""

    # 1. Fetch from configured news sources via NewsService
    try:
        news_svc = NewsService(league_id)

        # Default sources: HoopsHype (rumors/salary with actual written content)
        # ESPN excluded by default - player pages are stats, not news articles
        if sources is None:
            sources = ["hoopshype"]

        # Get player info from ID map for slug (still useful for name lookup)
        player_info = news_svc._get_player_info("espn", str(player_id))
        player_slug = None
        if player_info:
            player_slug = player_info.get("slug")
            if not name:
                name = player_info.get("name", "")

        # Fetch player page
        fetch_results = news_svc.fetch_player(
            str(player_id),
            sources=sources,
            patterns=["player"],
            player_slug=player_slug,
            parallel=True,
        )

        for r in fetch_results.successful:
            if r.content and len(r.content.strip()) >= _MIN_EXTRACT_CHARS:
                results.append({
                    "source": r.source,
                    "content": r.content,
                    "metadata": {
                        "title": f"{name or player_id} - {r.metadata.get('pattern', 'page')}",
                        "link": r.url,
                        "pattern": r.metadata.get("pattern"),
                        "fetched_via": "news_service",
                    },
                })
    except Exception as e:
        logger.warning(f"NewsService fetch failed for player {player_id}: {e}")

    # 2. If we need more results or web search requested, use SERP API
    if include_web_search or len(results) < num_results:
        # Get sport context from league config for more accurate searches
        # This is especially important for college sports where schools have multiple sports programs
        sport_context = ""
        league_display = "NBA"
        try:
            league_cfg = load_league_config(league_id)
            sport = league_cfg.raw.get("meta", {}).get("sport", "")
            display_name = league_cfg.raw.get("meta", {}).get("display_name", "")
            league_display = display_name or league_id.upper()
            # For college sports, add sport to disambiguate from football/other sports
            if "college" in display_name.lower() or league_id in ("cbb", "cfb", "cws"):
                sport_context = f" {sport}" if sport else " basketball"
            elif league_id not in ("nba", "nfl", "mlb", "nhl"):
                # For other non-major leagues, also include sport context
                sport_context = f" {sport}" if sport else ""
        except Exception:
            pass

        # Require at least one name token to appear in extracted content
        if not name or re.fullmatch(r"\d+", str(name).strip() or ""):
            name_tokens: List[str] = []
            q = f"{league_display} player {player_id}{sport_context} news".strip()
        else:
            name_tokens = [t for t in re.split(r"\s+", str(name).strip()) if len(t) >= 3]
            q = f"{name}{sport_context} news and latest".strip()

        web_results = _news_search(
            q,
            force_refresh=force_refresh,
            num_results=max(1, num_results - len(results)),
            league_id=league_id,
            required_terms=name_tokens,
        )
        # Add web results, avoiding duplicates
        existing_links = {r.get("metadata", {}).get("link") for r in results}
        for wr in web_results:
            link = wr.get("metadata", {}).get("link")
            if link not in existing_links:
                results.append(wr)
                existing_links.add(link)

    _cache_set(cache_key, results)
    return results

