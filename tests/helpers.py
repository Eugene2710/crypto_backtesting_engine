"""
helpers.py
----------
Shared test utility functions — not fixtures.

Fixtures are injected by pytest automatically from conftest.py.
Plain helper functions that are called directly with custom arguments
live here so they can be explicitly imported by any test module.
"""

from src.backtester.events import MarketEvent


def make_bar(
    bar_index: int,
    open_:     float,
    high:      float,
    low:       float,
    close:     float,
    volume:    float = 1_000.0,
    timestamp: int | None = None,
) -> MarketEvent:
    """
    Construct a synthetic MarketEvent for use in tests.

    Timestamp defaults to bar_index × 86,400,000 ms (one bar = one day)
    giving a realistic ascending series without needing real dates.
    """
    ts: int = timestamp if timestamp is not None else bar_index * 86_400_000
    return MarketEvent(
        symbol="BTCUSDT",
        timestamp=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bar_index=bar_index,
    )
