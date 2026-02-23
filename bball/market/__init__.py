"""
Market data integration for prediction markets (Kalshi, etc.)

This module provides:
- KalshiPublicClient: Unauthenticated client for reading market data
- MarketConnector: Authenticated client for trading (requires API keys)
- get_game_market_data(): High-level function to get market data for a game
"""


def __getattr__(name):
    if name in ("KalshiPublicClient", "get_game_market_data", "build_event_ticker"):
        from .kalshi import KalshiPublicClient, get_game_market_data, build_event_ticker
        return {"KalshiPublicClient": KalshiPublicClient,
                "get_game_market_data": get_game_market_data,
                "build_event_ticker": build_event_ticker}[name]
    if name == "MarketConnector":
        from .connector import MarketConnector
        return MarketConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KalshiPublicClient",
    "MarketConnector",
    "get_game_market_data",
    "build_event_ticker",
]
