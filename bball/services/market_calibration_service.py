"""
Market calibration service — re-exports from sportscore.

The implementation now lives in sportscore since it's sport-agnostic.
This module preserves backward compatibility for existing callers.
"""

from sportscore.services.market_calibration_service import (
    compute_and_store_market_calibration,
)

__all__ = ["compute_and_store_market_calibration"]
