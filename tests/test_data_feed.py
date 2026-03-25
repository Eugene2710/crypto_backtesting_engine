"""
test_data_feed.py
-----------------
Phase 1 integration tests — BinanceDataFeed.

These are true integration tests: they hit the live Binance public API and
validate the shape and integrity of the returned data. They will fail if
the network is unavailable or Binance changes their response format.

Tests in this file share one API response via the module-scoped
`btcusdt_daily_feed` fixture. Each test calls feed.reset() in its Arrange
section to restore the cursor to bar 0 before iterating.

Principles applied
------------------
- One assertion per test: each test verifies exactly one correctness property.
- Descriptive names: the test name states what property is being verified
  and what the expected outcome is.
- AAA structure: Arrange / Act / Assert sections are clearly separated.
"""

import math

import pytest

from src.backtester.data.binance import BinanceDataFeed
from src.backtester.events import MarketEvent


class TestBinanceDataFeed:

    def test_load_returns_exactly_500_bars(
        self, btcusdt_daily_feed: BinanceDataFeed
    ) -> None:
        """
        The feed must contain exactly the number of bars requested.
        Fewer bars would indicate a truncated API response or a rate-limit.
        """
        # Arrange
        feed = btcusdt_daily_feed

        # Act + Assert
        assert len(feed) == 500, (
            f"Expected 500 bars, got {len(feed)}. "
            "Binance may have truncated the response."
        )

    def test_bars_are_in_strictly_ascending_chronological_order(
        self, btcusdt_daily_feed: BinanceDataFeed
    ) -> None:
        """
        Timestamps must be strictly increasing — no duplicate or reversed bars.
        Out-of-order bars would cause the engine to process historical data
        as if it were current, producing lookahead bias.
        """
        # Arrange
        feed = btcusdt_daily_feed
        feed.reset()

        # Act
        timestamps: list[int] = [feed.get_bar(i).timestamp for i in range(len(feed))]

        # Assert
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], (
                f"Bar {i} timestamp {timestamps[i]} is not after "
                f"bar {i-1} timestamp {timestamps[i - 1]}."
            )

    def test_all_price_fields_are_positive_and_finite(
        self, btcusdt_daily_feed: BinanceDataFeed
    ) -> None:
        """
        Every OHLCV field must be a finite positive number.
        Zero or infinite values indicate a parsing failure or a bad API response.
        """
        # Arrange
        feed = btcusdt_daily_feed
        feed.reset()

        # Act + Assert
        for i in range(len(feed)):
            bar: MarketEvent = feed.get_bar(i)
            for field_name, value in [
                ("open",   bar.open),
                ("high",   bar.high),
                ("low",    bar.low),
                ("close",  bar.close),
                ("volume", bar.volume),
            ]:
                assert math.isfinite(value) and value > 0, (
                    f"Bar {i} field '{field_name}' is invalid: {value}"
                )

    def test_high_is_greater_than_or_equal_to_low_on_every_bar(
        self, btcusdt_daily_feed: BinanceDataFeed
    ) -> None:
        """
        High must always be >= low — a fundamental OHLCV integrity constraint.
        A bar where high < low is physically impossible and indicates corrupt data.
        """
        # Arrange
        feed = btcusdt_daily_feed
        feed.reset()

        # Act + Assert
        for i in range(len(feed)):
            bar: MarketEvent = feed.get_bar(i)
            assert bar.high >= bar.low, (
                f"Bar {i}: high ({bar.high}) < low ({bar.low}) — invalid OHLCV data."
            )

    def test_next_bar_iterates_all_bars_and_returns_none_at_end(
        self, btcusdt_daily_feed: BinanceDataFeed
    ) -> None:
        """
        next_bar() must yield exactly 500 bars then return None.
        Returning None too early would silently truncate the backtest.
        """
        # Arrange
        feed = btcusdt_daily_feed
        feed.reset()

        # Act
        count: int = 0
        while feed.next_bar() is not None:
            count += 1

        terminal: MarketEvent | None = feed.next_bar()

        # Assert
        assert count == 500
        assert terminal is None, "Expected None after last bar, got a MarketEvent."

    def test_reset_restores_cursor_to_bar_zero(
        self, btcusdt_daily_feed: BinanceDataFeed
    ) -> None:
        """
        reset() must allow the feed to be replayed from bar 0 without
        re-fetching from the network. This is required by walk-forward testing.
        """
        # Arrange
        feed = btcusdt_daily_feed
        feed.reset()
        first_bar_before: MarketEvent | None = feed.next_bar()

        # Act
        feed.reset()
        first_bar_after: MarketEvent | None = feed.next_bar()

        # Assert
        assert first_bar_before is not None
        assert first_bar_after is not None
        assert first_bar_before.timestamp == first_bar_after.timestamp, (
            "reset() did not restore the cursor — first bar timestamps differ."
        )
