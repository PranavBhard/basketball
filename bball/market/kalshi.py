"""
Kalshi prediction market integration.

Provides unauthenticated access to market data for display purposes.
Uses the public API at api.elections.kalshi.com (no auth required for reads).
"""

import time
import logging
from datetime import datetime, date
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from threading import Lock

import requests

from bball.league_config import load_league_config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL."""
    data: Any
    expires_at: float


class SimpleCache:
    """Thread-safe in-memory cache with TTL."""

    def __init__(self, default_ttl: int = 60):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.time() > entry.expires_at:
                del self._cache[key]
                return None
            return entry.data

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with TTL (seconds)."""
        ttl = ttl or self.default_ttl
        with self._lock:
            self._cache[key] = CacheEntry(
                data=value,
                expires_at=time.time() + ttl
            )

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()


# Global cache instance (60 second TTL by default)
_market_cache = SimpleCache(default_ttl=60)


class KalshiPublicClient:
    """
    Unauthenticated client for Kalshi public market data.

    Uses the public API which doesn't require authentication for
    reading market data, events, and series information.
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request to public API with retry on 429."""
        import time as _time
        url = f"{self.BASE_URL}{path}"
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 429 and attempt < max_retries:
                    wait = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
                    _time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries and hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                    _time.sleep(0.5 * (2 ** attempt))
                    continue
                logger.error(f"Kalshi API error: {e}")
                raise

    def get_event(self, event_ticker: str) -> Dict[str, Any]:
        """
        Get event details including all markets.

        Returns dict with 'event' and 'markets' keys.
        """
        return self._get(f"/events/{event_ticker}")

    def get_events(
        self,
        series_ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get events, optionally filtered by series."""
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        return self._get("/events", params)

    def get_market(self, ticker: str) -> Dict[str, Any]:
        """Get single market details."""
        return self._get(f"/markets/{ticker}")

    def get_markets(
        self,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get markets, optionally filtered."""
        params = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets", params)


def get_team_abbrev_map(league_id: str = "nba") -> Dict[str, str]:
    """
    Get mapping from Kalshi abbreviations to internal DB abbreviations.

    Returns dict where keys are Kalshi abbrevs, values are internal abbrevs.
    Only includes teams that differ; unlisted teams use same abbrev on both.
    """
    try:
        league = load_league_config(league_id)
        market_config = league.raw.get("market", {})
        return market_config.get("team_abbrev_map", {})
    except Exception as e:
        logger.warning(f"Could not load team abbrev map: {e}")
        return {}


def get_reverse_abbrev_map(league_id: str = "nba") -> Dict[str, str]:
    """
    Get mapping from internal DB abbreviations to Kalshi abbreviations.

    Returns dict where keys are internal abbrevs, values are Kalshi abbrevs.
    """
    forward_map = get_team_abbrev_map(league_id)
    return {v: k for k, v in forward_map.items()}


def internal_to_kalshi_abbrev(internal_abbrev: str, league_id: str = "nba") -> str:
    """Convert internal DB abbreviation to Kalshi abbreviation."""
    reverse_map = get_reverse_abbrev_map(league_id)
    return reverse_map.get(internal_abbrev, internal_abbrev)


def kalshi_to_internal_abbrev(kalshi_abbrev: str, league_id: str = "nba") -> str:
    """Convert Kalshi abbreviation to internal DB abbreviation."""
    abbrev_map = get_team_abbrev_map(league_id)
    return abbrev_map.get(kalshi_abbrev, kalshi_abbrev)


def build_event_ticker(
    game_date: date,
    away_team: str,
    home_team: str,
    league_id: str = "nba"
) -> str:
    """
    Build Kalshi event ticker from game info.

    Format: KXNBAGAME-{YY}{MON}{DD}{AWAY}{HOME}
    Example: KXNBAGAME-26JAN28SASHOU (San Antonio at Houston, Jan 28 2026)

    Args:
        game_date: Date of the game
        away_team: Internal DB abbreviation of away team
        home_team: Internal DB abbreviation of home team
        league_id: League identifier (default: "nba")

    Returns:
        Kalshi event ticker string
    """
    # Convert internal abbreviations to Kalshi format
    kalshi_away = internal_to_kalshi_abbrev(away_team, league_id)
    kalshi_home = internal_to_kalshi_abbrev(home_team, league_id)

    # Format date as YYMMMDD (e.g., 26JAN28)
    year_2digit = game_date.strftime("%y")
    month_abbrev = game_date.strftime("%b").upper()
    day_2digit = game_date.strftime("%d")

    # Get series ticker from league config
    try:
        league = load_league_config(league_id)
        series_ticker = league.raw.get("market", {}).get("series_ticker", "KXNBAGAME")
    except Exception:
        series_ticker = "KXNBAGAME"
        logger.warning(f"Could not load series_ticker from league config for '{league_id}', using default")

    return f"{series_ticker}-{year_2digit}{month_abbrev}{day_2digit}{kalshi_away}{kalshi_home}"


def parse_event_ticker(
    event_ticker: str,
    league_id: str = "nba",
) -> Optional[Dict[str, Any]]:
    """
    Parse Kalshi event ticker to extract game info.

    Handles both NBA (fixed 3-char abbrevs) and CBB (variable 2-4 char abbrevs).

    Args:
        event_ticker: e.g., "KXNBAGAME-26JAN28SASHOU" or "KXNCAAMBGAME-26FEB02KUTTU"
        league_id: League to use for abbreviation mapping

    Returns:
        Dict with 'date', 'away_team', 'home_team' (Kalshi abbrevs),
        'away_team_internal', 'home_team_internal' (converted to internal abbrevs)
        or None if invalid
    """
    try:
        # Split on dash to get date+teams part
        parts = event_ticker.split("-")
        if len(parts) < 2:
            return None

        game_part = parts[-1]  # e.g., "26JAN28SASHOU" or "26FEB02KUTTU"

        # Date is always first 7 chars: YY (2) + MON (3) + DD (2)
        if len(game_part) < 9:  # Need at least date + 2 chars for teams
            return None

        year_2digit = game_part[:2]
        month_abbrev = game_part[2:5]
        day_2digit = game_part[5:7]
        teams_part = game_part[7:]  # e.g., "SASHOU" or "KUTTU"

        # Parse date
        date_str = f"{month_abbrev} {day_2digit} 20{year_2digit}"
        game_date = datetime.strptime(date_str, "%b %d %Y").date()

        # For NBA (3-char fixed): split in half
        # For CBB (variable length): we need to try different split points
        away_team = None
        home_team = None

        if league_id == "nba":
            # NBA uses fixed 3-char abbreviations
            if len(teams_part) >= 6:
                away_team = teams_part[:3]
                home_team = teams_part[3:6]
        else:
            # CBB uses variable length (2-4 chars each)
            # Build set of all known Kalshi abbreviations
            abbrev_map = get_team_abbrev_map(league_id)
            all_kalshi_abbrevs = set(abbrev_map.keys())

            # Also add all ESPN abbreviations as valid (they often match)
            try:
                from bball.mongo import Mongo
                db = Mongo().db
                league = load_league_config(league_id)
                teams_coll = league.collections.get("teams", "cbb_teams")
                espn_abbrevs = set(db[teams_coll].distinct("abbreviation"))
                all_kalshi_abbrevs.update(espn_abbrevs)
            except Exception:
                pass

            # Find all valid splits and pick the best one
            valid_splits = []
            for away_len in range(2, min(5, len(teams_part) - 1)):
                candidate_away = teams_part[:away_len]
                candidate_home = teams_part[away_len:]

                if not (2 <= len(candidate_home) <= 4):
                    continue

                # Score the split: known abbrevs are better
                away_known = candidate_away in all_kalshi_abbrevs
                home_known = candidate_home in all_kalshi_abbrevs

                if away_known and home_known:
                    # Both known - best case
                    valid_splits.append((2, candidate_away, candidate_home))
                elif away_known or home_known:
                    # One known
                    valid_splits.append((1, candidate_away, candidate_home))
                elif candidate_away.isupper() and candidate_home.isupper():
                    # Both valid format
                    valid_splits.append((0, candidate_away, candidate_home))

            # Pick the best split (highest score, then prefer balanced lengths)
            if valid_splits:
                valid_splits.sort(key=lambda x: (x[0], -abs(len(x[1]) - len(x[2]))), reverse=True)
                _, away_team, home_team = valid_splits[0]

            # Fallback: split in half
            if not away_team or not home_team:
                mid = len(teams_part) // 2
                away_team = teams_part[:mid]
                home_team = teams_part[mid:]

        if not away_team or not home_team:
            return None

        # Convert to internal abbreviations
        away_internal = kalshi_to_internal_abbrev(away_team, league_id)
        home_internal = kalshi_to_internal_abbrev(home_team, league_id)

        return {
            "date": game_date,
            "away_team": away_team,
            "home_team": home_team,
            "away_team_internal": away_internal,
            "home_team_internal": home_internal,
            "series_ticker": "-".join(parts[:-1]),
        }
    except Exception as e:
        logger.debug(f"Could not parse event ticker {event_ticker}: {e}")
        return None


@dataclass
class MarketData:
    """Normalized market data for a game."""
    event_ticker: str
    home_team: str  # Internal DB abbreviation
    away_team: str  # Internal DB abbreviation
    home_yes_price: float  # Price in dollars (0.00 - 1.00)
    home_yes_bid: float
    home_yes_ask: float
    away_yes_price: float
    away_yes_bid: float
    away_yes_ask: float
    home_volume: int
    away_volume: int
    total_liquidity: float  # In dollars
    status: str  # "active", "closed", "settled", etc.
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "event_ticker": self.event_ticker,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_yes_price": self.home_yes_price,
            "home_yes_bid": self.home_yes_bid,
            "home_yes_ask": self.home_yes_ask,
            "away_yes_price": self.away_yes_price,
            "away_yes_bid": self.away_yes_bid,
            "away_yes_ask": self.away_yes_ask,
            "home_volume": self.home_volume,
            "away_volume": self.away_volume,
            "total_liquidity": self.total_liquidity,
            "status": self.status,
            "last_updated": self.last_updated.isoformat(),
        }


def get_game_market_data(
    game_date: date,
    away_team: str,
    home_team: str,
    league_id: str = "nba",
    use_cache: bool = True,
    cache_ttl: int = 60
) -> Optional[MarketData]:
    """
    Get market data for a specific game.

    Args:
        game_date: Date of the game
        away_team: Internal DB abbreviation of away team
        home_team: Internal DB abbreviation of home team
        league_id: League identifier
        use_cache: Whether to use cached data
        cache_ttl: Cache TTL in seconds

    Returns:
        MarketData object or None if not found/error
    """
    event_ticker = build_event_ticker(game_date, away_team, home_team, league_id)
    cache_key = f"market:{event_ticker}"

    # Check cache
    if use_cache:
        cached = _market_cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {event_ticker}")
            return cached

    # Fetch from API
    client = KalshiPublicClient()
    try:
        data = client.get_event(event_ticker)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            logger.debug(f"No market found for {event_ticker}")
            return None
        logger.error(f"API error fetching {event_ticker}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching market data for {event_ticker}: {e}")
        return None

    # Parse response
    markets = data.get("markets", [])
    if not markets:
        logger.debug(f"No markets in response for {event_ticker}")
        return None

    # Convert Kalshi abbrevs to internal
    kalshi_home = internal_to_kalshi_abbrev(home_team, league_id)
    kalshi_away = internal_to_kalshi_abbrev(away_team, league_id)

    home_market = None
    away_market = None

    for market in markets:
        ticker = market.get("ticker", "")
        # Market ticker ends with team abbrev: KXNBAGAME-26JAN28SASHOU-HOU
        if ticker.endswith(f"-{kalshi_home}"):
            home_market = market
        elif ticker.endswith(f"-{kalshi_away}"):
            away_market = market

    if not home_market and not away_market:
        logger.warning(f"Could not identify team markets for {event_ticker}")
        return None

    def extract_market_data(m: Optional[Dict]) -> Tuple[float, float, float, int, str]:
        """Extract (mid_price, bid, ask, volume, status) from market dict."""
        if not m:
            return (0.0, 0.0, 0.0, 0, "unknown")
        # Prices are in cents (0-100), convert to dollars
        yes_bid = m.get("yes_bid", 0) / 100.0
        yes_ask = m.get("yes_ask", 0) / 100.0
        # Use mid-market price for more accurate live odds;
        # fall back to last_price if bid/ask unavailable
        if yes_bid > 0 and yes_ask > 0:
            mid_price = (yes_bid + yes_ask) / 2.0
        else:
            mid_price = m.get("last_price", 0) / 100.0
        volume = m.get("volume", 0)
        status = m.get("status", "unknown")
        return (mid_price, yes_bid, yes_ask, volume, status)

    home_price, home_bid, home_ask, home_vol, home_status = extract_market_data(home_market)
    away_price, away_bid, away_ask, away_vol, away_status = extract_market_data(away_market)

    # Calculate total liquidity
    home_liq = float(home_market.get("liquidity_dollars", "0")) if home_market else 0
    away_liq = float(away_market.get("liquidity_dollars", "0")) if away_market else 0
    total_liquidity = home_liq + away_liq

    # Use home market status as primary (they should be the same)
    status = home_status if home_market else away_status

    result = MarketData(
        event_ticker=event_ticker,
        home_team=home_team,
        away_team=away_team,
        home_yes_price=home_price,
        home_yes_bid=home_bid,
        home_yes_ask=home_ask,
        away_yes_price=away_price,
        away_yes_bid=away_bid,
        away_yes_ask=away_ask,
        home_volume=home_vol,
        away_volume=away_vol,
        total_liquidity=total_liquidity,
        status=status,
        last_updated=datetime.utcnow(),
    )

    # Cache result
    if use_cache:
        _market_cache.set(cache_key, result, cache_ttl)

    return result


def clear_market_cache() -> None:
    """Clear the market data cache."""
    _market_cache.clear()


def find_game_from_event_ticker(
    event_ticker: str,
    league_id: str = "nba",
    db=None,
) -> Optional[Dict[str, Any]]:
    """
    Look up a game in the database from a Kalshi event ticker.

    Args:
        event_ticker: Kalshi event ticker (e.g., "KXNBAGAME-26JAN28SASHOU")
        league_id: League identifier
        db: MongoDB database (if None, creates new connection)

    Returns:
        Game document from database, or None if not found
    """
    from bball.mongo import Mongo

    # Parse the event ticker
    parsed = parse_event_ticker(event_ticker, league_id)
    if not parsed:
        logger.warning(f"Could not parse event ticker: {event_ticker}")
        return None

    game_date = parsed["date"]
    away_internal = parsed["away_team_internal"]
    home_internal = parsed["home_team_internal"]

    # Get database
    if db is None:
        db = Mongo().db

    # Get league config for collection name
    try:
        league = load_league_config(league_id)
        games_collection = league.collections.get("games", "stats_nba")
    except Exception:
        games_collection = f"{league_id}_stats_games" if league_id != "nba" else "stats_nba"

    # Format date for query
    date_str = game_date.strftime("%Y-%m-%d")

    # Query for the game
    # Try multiple matching strategies
    game = None

    # Strategy 1: Match by team abbreviations
    game = db[games_collection].find_one({
        "date": date_str,
        "awayTeam.name": away_internal,
        "homeTeam.name": home_internal,
    })

    # Strategy 2: Try with case-insensitive matching
    if not game:
        game = db[games_collection].find_one({
            "date": date_str,
            "awayTeam.name": {"$regex": f"^{away_internal}$", "$options": "i"},
            "homeTeam.name": {"$regex": f"^{home_internal}$", "$options": "i"},
        })

    # Strategy 3: For leagues using team_id (CBB, WCBB), try matching by team_id
    if not game and league_id != "nba":
        # Look up team IDs
        teams_collection = league.collections.get("teams", f"{league_id}_teams")
        away_team_doc = db[teams_collection].find_one({"abbreviation": away_internal})
        home_team_doc = db[teams_collection].find_one({"abbreviation": home_internal})

        if away_team_doc and home_team_doc:
            game = db[games_collection].find_one({
                "date": date_str,
                "awayTeam.team_id": str(away_team_doc.get("team_id")),
                "homeTeam.team_id": str(home_team_doc.get("team_id")),
            })

    if game:
        logger.debug(f"Found game {game.get('game_id')} for {event_ticker}")
    else:
        logger.debug(f"No game found for {event_ticker} ({away_internal} @ {home_internal} on {date_str})")

    return game


def get_kalshi_events_for_date(
    game_date: date,
    league_id: str = "nba",
    use_cache: bool = True,
    cache_ttl: int = 300,
) -> list:
    """
    Get all Kalshi events for a specific date.

    Args:
        game_date: Date to fetch events for
        league_id: League identifier
        use_cache: Whether to use cached data
        cache_ttl: Cache TTL in seconds

    Returns:
        List of event dicts from Kalshi API
    """
    cache_key = f"events:{league_id}:{game_date.isoformat()}"

    if use_cache:
        cached = _market_cache.get(cache_key)
        if cached is not None:
            return cached

    # Get series ticker from config
    try:
        league = load_league_config(league_id)
        series_ticker = league.raw.get("market", {}).get("series_ticker", "KXNBAGAME")
    except Exception:
        series_ticker = "KXNBAGAME"
        logger.warning(f"Could not load series_ticker from league config for '{league_id}', using default")

    client = KalshiPublicClient()
    all_events = []

    try:
        # Fetch events for the series
        data = client.get_events(series_ticker=series_ticker, limit=200)
        events = data.get("events", [])

        # Filter by date
        date_prefix = f"{game_date.strftime('%y')}{game_date.strftime('%b').upper()}{game_date.strftime('%d')}"

        for event in events:
            ticker = event.get("event_ticker", "")
            # Check if this event is for the target date
            if f"-{date_prefix}" in ticker:
                all_events.append(event)

    except Exception as e:
        logger.error(f"Error fetching Kalshi events for {game_date}: {e}")

    if use_cache and all_events:
        _market_cache.set(cache_key, all_events, cache_ttl)

    return all_events


@dataclass
class PortfolioMatch:
    """Result of matching portfolio items to games."""
    game_data: Dict[str, Dict[str, Any]]  # game_id -> {positions, fills, orders, parlay_fills}
    unmatched_fills: list  # Fills that couldn't be matched to single games
    debug_info: Optional[Dict[str, Any]] = None  # Debug info for troubleshooting


def match_portfolio_to_games(
    games: list,
    positions: list,
    fills: list,
    orders: list,
    settlements: list,
    game_date: date,
    league_id: str = "nba",
) -> PortfolioMatch:
    """
    Match Kalshi portfolio items (positions, fills, orders) to games.

    This is the core logic for attributing trading activity to specific games.
    Handles both moneyline/winner markets and spread markets.

    Args:
        games: List of game dicts with game_id, homeTeam, awayTeam
        positions: Raw positions from Kalshi API
        fills: Raw fills from Kalshi API
        orders: Raw orders from Kalshi API
        settlements: Raw settlements from Kalshi API
        game_date: Date of the games
        league_id: League identifier

    Returns:
        PortfolioMatch with game_data dict and unmatched_fills list
    """
    # Get series tickers from league config
    try:
        league = load_league_config(league_id)
        series_ticker = league.raw.get("market", {}).get("series_ticker", "KXNBAGAME")
        spread_series_ticker = league.raw.get("market", {}).get("spread_series_ticker", "KXNBASPREAD")
    except Exception:
        series_ticker = "KXNBAGAME"
        spread_series_ticker = "KXNBASPREAD"
        logger.warning(f"Could not load series/spread tickers from league config for '{league_id}', using defaults")

    # Build event tickers for all games (both winner and spread tickers)
    game_tickers = {}  # event_ticker -> game_id (for winner markets)
    spread_tickers = {}  # spread_event_ticker -> game_id (for spread markets)

    for game in games:
        game_id = game.get('game_id')
        home_team = game.get('homeTeam', {}).get('name')
        away_team = game.get('awayTeam', {}).get('name')
        if game_id and home_team and away_team:
            event_ticker = build_event_ticker(game_date, away_team, home_team, league_id)
            game_tickers[event_ticker] = game_id
            # Also build spread event ticker (same format but different prefix)
            if spread_series_ticker:
                spread_event = event_ticker.replace(series_ticker, spread_series_ticker, 1)
                spread_tickers[spread_event] = game_id
                logger.debug(f"[SPREAD DEBUG] Built spread ticker: {spread_event} -> {game_id}")

    logger.info(f"[SPREAD DEBUG] Built {len(spread_tickers)} spread tickers for {len(games)} games")

    # Build settlement lookup for P&L calculation
    settlement_lookup = {}
    for settlement in settlements:
        ticker = settlement.get('ticker', '')
        revenue = settlement.get('revenue', 0)
        settlement_lookup[ticker] = {
            'settled': True,
            'revenue': revenue,
            'pnl': revenue - settlement.get('cost_basis', 0) if 'cost_basis' in settlement else revenue
        }

    # Initialize game_data structure
    game_data = {
        game.get('game_id'): {
            'positions': [],
            'fills': [],
            'orders': [],
            'parlay_fills': []
        }
        for game in games if game.get('game_id')
    }

    # Build team abbreviation mappings for parlay detection
    reverse_abbrev_map = get_reverse_abbrev_map(league_id)  # internal -> kalshi
    team_to_game = {}
    for game in games:
        game_id = game.get('game_id')
        home_team = game.get('homeTeam', {}).get('name')
        away_team = game.get('awayTeam', {}).get('name')
        if game_id:
            if home_team:
                team_to_game[home_team.upper()] = game_id
                kalshi_home = reverse_abbrev_map.get(home_team.upper(), home_team.upper())
                if kalshi_home != home_team.upper():
                    team_to_game[kalshi_home] = game_id
            if away_team:
                team_to_game[away_team.upper()] = game_id
                kalshi_away = reverse_abbrev_map.get(away_team.upper(), away_team.upper())
                if kalshi_away != away_team.upper():
                    team_to_game[kalshi_away] = game_id

    # Match positions to games
    for pos in positions:
        ticker = pos.get('ticker', '')
        event_ticker = pos.get('event_ticker', '')
        matched = False
        game_id = None
        team = ''
        spread_value = None
        is_spread = False

        # Try to match as a winner/moneyline position first
        if event_ticker in game_tickers:
            matched = True
            game_id = game_tickers[event_ticker]
            team = ticker.split('-')[-1] if '-' in ticker else ''

        # Try to match as a spread position if not matched
        if not matched and event_ticker in spread_tickers:
            matched = True
            game_id = spread_tickers[event_ticker]
            is_spread = True
            ticker_parts = ticker.split('-')

            # Handle format: KXNBASPREAD-26FEB01OKCDEN-OKC7 (team+spread combined)
            import re
            last_part = ticker_parts[-1] if ticker_parts else ''
            match = re.match(r'^([A-Z]+)([\d.]+)$', last_part)
            if match:
                team = match.group(1)
                spread_value = match.group(2)
            elif len(ticker_parts) >= 4:
                spread_value = ticker_parts[-1]
                team = ticker_parts[-2]
            else:
                team = last_part
                spread_value = None

        if matched and game_id and game_id in game_data:
            count = pos.get('total_traded', 0)
            avg_price = pos.get('average_price_cents', 0) / 100.0
            market_value = pos.get('market_value_cents', 0) / 100.0

            # Check if settled
            settlement = settlement_lookup.get(ticker, {})
            is_settled = settlement.get('settled', False)

            # Cost basis for P&L calculation
            cost_basis = count * avg_price

            # Determine status and P&L
            if is_settled:
                pnl_cents = settlement.get('pnl', 0)
                status = 'won' if pnl_cents > 0 else 'lost'
                pnl = pnl_cents / 100.0
            else:
                status = 'live'
                pnl = market_value - cost_basis

            # Parse market label (spread shows as "TEAM -X.X" format)
            if is_spread and spread_value:
                market_label = f"{team} -{spread_value}"
            else:
                ticker_parts = ticker.split('-')
                if len(ticker_parts) > 3:
                    market_label = f"{team} -{ticker_parts[-1]}"
                else:
                    market_label = team

            game_data[game_id]['positions'].append({
                'ticker': ticker,
                'team': team,
                'market_label': market_label,
                'side': pos.get('side', ''),
                'action': 'buy',
                'count': count,
                'avg_price': avg_price,
                'current_value': market_value,
                'cost': count * avg_price,
                'pnl': pnl,
                'status': status,
                'is_spread': is_spread,
                'spread_value': spread_value
            })

    # Match fills to games
    unmatched_fills = []

    for fill in fills:
        ticker = fill.get('ticker', '')
        fill_event_ticker = fill.get('event_ticker', '')  # Kalshi may provide this directly
        matched = False
        game_id = None
        team = ''
        spread_value = None

        # Try to match using event_ticker field if available (most reliable)
        if fill_event_ticker:
            if fill_event_ticker in game_tickers:
                matched = True
                game_id = game_tickers[fill_event_ticker]
                team = ticker.split('-')[-1] if '-' in ticker else ''
            elif fill_event_ticker in spread_tickers:
                matched = True
                game_id = spread_tickers[fill_event_ticker]
                # Parse spread details from ticker
                parts = ticker.split('-')
                if len(parts) >= 4:
                    spread_value = parts[-1]
                    team = parts[-2]
                else:
                    team = parts[-1] if parts else ''
                logger.info(f"[SPREAD DEBUG] MATCHED via event_ticker: {ticker} -> game {game_id}")

        # Fallback: Try to match as a winner/moneyline fill by parsing ticker
        if not matched:
            parts = ticker.rsplit('-', 1)
            if len(parts) == 2:
                event_ticker = parts[0]
                team = parts[1]
                if event_ticker in game_tickers:
                    matched = True
                    game_id = game_tickers[event_ticker]

        # Fallback: Try to match as a spread fill by parsing ticker
        # Format: KXNBASPREAD-26FEB01OKCDEN-OKC7 (team+spread combined in last part)
        if not matched and spread_series_ticker and ticker.startswith(spread_series_ticker):
            parts = ticker.split('-')
            logger.debug(f"[SPREAD DEBUG] Trying to match spread fill: {ticker}, parts={parts}")

            # Handle both formats:
            # Old expected: KXNBASPREAD-26FEB01OKCDEN-OKC-7 (4 parts)
            # Actual: KXNBASPREAD-26FEB01OKCDEN-OKC7 (3 parts, team+spread combined)
            if len(parts) >= 3:
                last_part = parts[-1]  # e.g., "OKC7" or "MEM9" or "DEN5.5"

                # Extract team (letters) and spread (numbers) from combined string
                import re
                match = re.match(r'^([A-Z]+)([\d.]+)$', last_part)
                if match:
                    team = match.group(1)  # e.g., "OKC"
                    spread_value = match.group(2)  # e.g., "7"
                    spread_event = '-'.join(parts[:-1])  # e.g., "KXNBASPREAD-26FEB01OKCDEN"
                elif len(parts) >= 4:
                    # Fallback to old format if present
                    spread_value = parts[-1]
                    team = parts[-2]
                    spread_event = '-'.join(parts[:-2])
                else:
                    spread_event = '-'.join(parts[:-1])
                    team = last_part
                    spread_value = None

                logger.debug(f"[SPREAD DEBUG] Parsed: team={team}, spread={spread_value}, spread_event={spread_event}")
                logger.debug(f"[SPREAD DEBUG] Looking for spread_event in spread_tickers (has {len(spread_tickers)} entries)")

                if spread_event in spread_tickers:
                    matched = True
                    game_id = spread_tickers[spread_event]
                    logger.info(f"[SPREAD DEBUG] MATCHED spread fill {ticker} to game {game_id}")
                else:
                    logger.warning(f"[SPREAD DEBUG] spread_event {spread_event} NOT in spread_tickers. Available: {list(spread_tickers.keys())[:5]}")

        if matched and game_id and game_id in game_data:
            side = fill.get('side', '')
            price_cents = fill.get('yes_price', 0) if side == 'yes' else fill.get('no_price', 0)
            count = fill.get('count', 0)
            action = fill.get('action', '')

            # Check if settled
            settlement = settlement_lookup.get(ticker, {})
            is_settled = settlement.get('settled', False)

            if is_settled:
                revenue_cents = settlement.get('revenue', 0)
                status = 'won' if revenue_cents > 0 else 'lost'
            else:
                status = 'live'

            # Parse market label (spread shows as "TEAM -X.X" format)
            if spread_value:
                market_label = f"{team} -{spread_value}"
            else:
                market_label = team

            game_data[game_id]['fills'].append({
                'ticker': ticker,
                'team': team,
                'market_label': market_label,
                'side': side,
                'action': action,
                'count': count,
                'price': price_cents / 100.0,
                'cost': count * (price_cents / 100.0),
                'time': fill.get('created_time', ''),
                'status': status,
                'is_spread': spread_value is not None,
                'spread_value': spread_value
            })
        else:
            unmatched_fills.append(fill)

    # Process unmatched fills as potential parlays
    abbrev_map = get_team_abbrev_map(league_id)

    for fill in unmatched_fills:
        ticker = fill.get('ticker', '')

        # Parlay tickers contain multiple team abbreviations
        involved_game_ids = set()
        involved_teams = []

        for part in ticker.split('-'):
            team_abbrev = part.upper()
            internal_abbrev = abbrev_map.get(team_abbrev, team_abbrev)

            if internal_abbrev in team_to_game:
                involved_game_ids.add(team_to_game[internal_abbrev])
                involved_teams.append(internal_abbrev)
            elif team_abbrev in team_to_game:
                involved_game_ids.add(team_to_game[team_abbrev])
                involved_teams.append(team_abbrev)

        # If matches multiple games, it's a parlay
        if len(involved_game_ids) >= 2:
            side = fill.get('side', '')
            price_cents = fill.get('yes_price', 0) if side == 'yes' else fill.get('no_price', 0)
            count = fill.get('count', 0)
            action = fill.get('action', '')
            cost = count * (price_cents / 100.0)

            # Check if settled
            settlement = settlement_lookup.get(ticker, {})
            is_settled = settlement.get('settled', False)

            if is_settled:
                pnl_cents = settlement.get('pnl', 0)
                status = 'won' if pnl_cents > 0 else 'lost'
                pnl = pnl_cents / 100.0
                # If lost and pnl is 0 (settlement missing cost_basis), use negative cost
                if status == 'lost' and pnl == 0:
                    pnl = -cost
            else:
                status = 'live'
                pnl = 0

            num_legs = len(involved_game_ids)
            fractional_cost = cost / num_legs
            fractional_pnl = pnl / num_legs

            # Add to all involved games
            for gid in involved_game_ids:
                if gid in game_data:
                    game_data[gid]['parlay_fills'].append({
                        'ticker': ticker,
                        'teams': involved_teams,
                        'all_game_ids': list(involved_game_ids),
                        'market_label': f"{num_legs}-leg parlay",
                        'side': side,
                        'action': action,
                        'count': count,
                        'price': price_cents / 100.0,
                        'total_cost': cost,
                        'fractional_cost': fractional_cost,
                        'total_pnl': pnl,
                        'fractional_pnl': fractional_pnl,
                        'num_legs': num_legs,
                        'time': fill.get('created_time', ''),
                        'status': status,
                        'is_parlay': True
                    })

    # Process MULTIGAME combo fills (need to fetch market details to get legs)
    public_client = KalshiPublicClient()
    processed_combo_tickers = set()  # Avoid duplicate processing

    # Collect remaining unmatched fills that are MULTIGAME combos
    combo_fills = [f for f in unmatched_fills if 'MULTIGAME' in f.get('ticker', '').upper()]
    logger.info(f"[COMBO DEBUG] Found {len(combo_fills)} MULTIGAME combo fills to process")

    for fill in combo_fills:
        ticker = fill.get('ticker', '')

        # Skip if already processed (same combo appears multiple times)
        if ticker in processed_combo_tickers:
            continue
        processed_combo_tickers.add(ticker)

        try:
            # Fetch market details to get the legs
            market_resp = public_client.get_market(ticker)
            market = market_resp.get('market', {})

            # Get legs from mve_selected_legs array
            legs = market.get('mve_selected_legs', [])
            if not legs:
                # Fallback: try custom_strike field
                custom_strike = market.get('custom_strike', {})
                associated_events = custom_strike.get('Associated Events', '')
                associated_sides = custom_strike.get('Associated Market Sides', '')
                if associated_events:
                    events_list = [e.strip() for e in associated_events.split(',')]
                    sides_list = [s.strip() for s in associated_sides.split(',')] if associated_sides else []
                    legs = [
                        {'event_ticker': ev, 'side': sides_list[i] if i < len(sides_list) else 'yes'}
                        for i, ev in enumerate(events_list)
                    ]

            if not legs:
                logger.warning(f"[COMBO DEBUG] No legs found for combo {ticker}")
                continue

            logger.info(f"[COMBO DEBUG] Combo {ticker} has {len(legs)} legs")

            # Match legs to games
            matched_game_ids = set()
            matched_teams = []
            leg_details = []

            for leg in legs:
                leg_event_ticker = leg.get('event_ticker', '')
                leg_side = leg.get('side', 'yes')
                leg_market_ticker = leg.get('market_ticker', '')

                # Try to match this leg to one of our games
                matched_game_id = None

                # Check if leg event ticker matches our game tickers
                if leg_event_ticker in game_tickers:
                    matched_game_id = game_tickers[leg_event_ticker]
                elif leg_event_ticker in spread_tickers:
                    matched_game_id = spread_tickers[leg_event_ticker]
                else:
                    # Try parsing event ticker to extract teams
                    parsed = parse_event_ticker(leg_event_ticker)
                    if parsed:
                        leg_date, away_team, home_team = parsed
                        # Check if date matches
                        if leg_date == game_date:
                            # Try to find game by teams
                            away_upper = away_team.upper()
                            home_upper = home_team.upper()
                            if away_upper in team_to_game:
                                matched_game_id = team_to_game[away_upper]
                            elif home_upper in team_to_game:
                                matched_game_id = team_to_game[home_upper]

                if matched_game_id:
                    matched_game_ids.add(matched_game_id)
                    # Extract team from market ticker (last part after -)
                    team = leg_market_ticker.split('-')[-1] if leg_market_ticker else ''
                    matched_teams.append(team)
                    leg_details.append({
                        'event_ticker': leg_event_ticker,
                        'side': leg_side,
                        'team': team,
                        'game_id': matched_game_id
                    })

            # Only add if at least one leg matches a game on this date
            if matched_game_ids:
                side = fill.get('side', '')
                price_cents = fill.get('yes_price', 0) if side == 'yes' else fill.get('no_price', 0)
                count = fill.get('count', 0)
                action = fill.get('action', '')
                cost = count * (price_cents / 100.0)

                # Check if settled
                settlement = settlement_lookup.get(ticker, {})
                is_settled = settlement.get('settled', False)

                if is_settled:
                    pnl_cents = settlement.get('pnl', 0)
                    status = 'won' if pnl_cents > 0 else 'lost'
                    pnl = pnl_cents / 100.0
                    # If lost and pnl is 0 (settlement missing cost_basis), use negative cost
                    if status == 'lost' and pnl == 0:
                        pnl = -cost
                else:
                    status = 'live'
                    pnl = 0

                num_legs = len(legs)
                num_matched = len(matched_game_ids)

                logger.info(f"[COMBO DEBUG] Combo {ticker}: {num_matched}/{num_legs} legs matched to games on this date")

                # Add to each matched game
                for gid in matched_game_ids:
                    if gid in game_data:
                        # Find this game's leg details
                        this_game_legs = [ld for ld in leg_details if ld.get('game_id') == gid]
                        this_game_team = this_game_legs[0].get('team', '') if this_game_legs else ''
                        this_game_side = this_game_legs[0].get('side', 'yes') if this_game_legs else side

                        game_data[gid]['parlay_fills'].append({
                            'ticker': ticker,
                            'teams': matched_teams,
                            'all_game_ids': list(matched_game_ids),
                            'this_game_team': this_game_team,
                            'this_game_side': this_game_side,
                            'market_label': f"{num_legs}-leg combo",
                            'side': side,
                            'action': action,
                            'count': count,
                            'price': price_cents / 100.0,
                            'total_cost': cost,
                            'total_pnl': pnl,
                            'num_legs': num_legs,
                            'num_matched_today': num_matched,
                            'time': fill.get('created_time', ''),
                            'status': status,
                            'is_parlay': True,
                            'is_combo': True,
                            'leg_details': leg_details
                        })

        except Exception as e:
            logger.warning(f"[COMBO DEBUG] Failed to process combo {ticker}: {e}")
            continue

    # Match orders to games
    for order in orders:
        ticker = order.get('ticker', '')
        event_ticker = order.get('event_ticker', '')
        matched = False
        game_id = None
        team = ''
        spread_value = None
        is_spread = False

        # Try to match as a winner/moneyline order first
        if event_ticker in game_tickers:
            matched = True
            game_id = game_tickers[event_ticker]
            ticker_parts = ticker.split('-')
            team = ticker_parts[-1] if len(ticker_parts) >= 3 else ''

        # Try to match as a spread order if not matched
        if not matched and event_ticker in spread_tickers:
            matched = True
            game_id = spread_tickers[event_ticker]
            is_spread = True
            ticker_parts = ticker.split('-')
            if len(ticker_parts) >= 4:
                spread_value = ticker_parts[-1]
                team = ticker_parts[-2]
            else:
                team = ticker_parts[-1] if ticker_parts else ''

        if matched and game_id and game_id in game_data:
            # Parse market label (spread shows as "TEAM -X.X" format)
            if is_spread and spread_value:
                market_label = f"{team} -{spread_value}"
            else:
                ticker_parts = ticker.split('-')
                if len(ticker_parts) > 3:
                    market_label = f"{ticker_parts[-2]} -{ticker_parts[-1]}"
                    team = ticker_parts[-2]
                else:
                    market_label = team

            count = order.get('remaining_count', 0)
            price = order.get('yes_price', 0) / 100.0 if order.get('yes_price') else order.get('no_price', 0) / 100.0

            game_data[game_id]['orders'].append({
                'ticker': ticker,
                'team': team,
                'market_label': market_label,
                'side': order.get('side', ''),
                'action': order.get('action', ''),
                'count': count,
                'price': price,
                'cost': count * price,
                'status': 'pending',
                'is_spread': is_spread,
                'spread_value': spread_value
            })

    # Build debug info
    debug_info = {
        'spread_tickers': list(spread_tickers.keys())[:10],  # First 10 for debugging
        'game_tickers': list(game_tickers.keys())[:10],
        'spread_series_ticker': spread_series_ticker,
        'series_ticker': series_ticker,
    }

    return PortfolioMatch(game_data=game_data, unmatched_fills=unmatched_fills, debug_info=debug_info)
