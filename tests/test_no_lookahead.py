"""
test_no_lookahead.py
--------------------
Phase 2 unit tests — zero lookahead guarantee.

The engine's most critical correctness requirement is that at bar N, the
strategy never has access to bar N+1 data. These tests verify this
structurally by instrumenting a recording strategy that asserts its own
invariant on every bar it receives.

These are unit tests: no network calls, no external dependencies.
All bars are constructed synthetically in memory.

Principles applied
------------------
- Tests use a purpose-built RecordingStrategy rather than a real strategy
  so the assertion is embedded in the component under test (the engine),
  not in the test body after the fact.
- Synthetic bars with predictable values make failures easy to diagnose.
- One concern per test: lookahead by bar index, and correct bar count.
"""
from tests.helpers import make_bar
from src.backtester.broker import SimulatedBroker
from src.backtester.engine import BacktestEngine
from src.backtester.events import MarketEvent, SignalEvent
from src.backtester.portfolio import Portfolio
from src.backtester.strategies.base import Strategy


# ---------------------------------------------------------------------------
# Recording strategy — embedded invariant assertion
# ---------------------------------------------------------------------------

class RecordingStrategy(Strategy):
    """
    A test-only strategy that records every bar index it receives and
    asserts that no future bar index has been seen yet.

    If the engine ever passes bar N+1 before bar N is finished processing,
    the assertion inside on_bar() will fail immediately with a clear message.
    """

    def __init__(self) -> None:
        self._seen_indices: list[int] = []

    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        # Assert no future index has been seen before the current one arrives.
        if self._seen_indices:
            max_seen: int = max(self._seen_indices)
            assert max_seen < event.bar_index or max_seen == event.bar_index, (
                f"Lookahead detected: saw bar {max_seen} before processing bar {event.bar_index}"
            )
        self._seen_indices.append(event.bar_index)
        return None

    def reset(self) -> None:
        self._seen_indices.clear()

    @property
    def seen_indices(self) -> list[int]:
        return self._seen_indices


# ---------------------------------------------------------------------------
# Synthetic feed helper
# ---------------------------------------------------------------------------

def build_static_feed(n_bars: int) -> list[MarketEvent]:
    """Build n_bars synthetic bars with strictly increasing bar_index."""
    return [
        make_bar(i, open_=100.0 + i, high=105.0 + i, low=95.0 + i, close=100.0 + i)
        for i in range(n_bars)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoLookahead:

    def test_strategy_only_receives_bars_in_order(
        self,
        broker:    SimulatedBroker,
        portfolio: Portfolio,
    ) -> None:
        """
        On every call to on_bar(bar_N), the strategy must not have seen
        any bar with index > N. The RecordingStrategy asserts this invariant
        internally on every bar — if lookahead occurs the test fails inside
        on_bar() with a descriptive message.
        """
        # Arrange
        from src.backtester.data.binance import BinanceDataFeed

        bars = build_static_feed(10)

        class StaticFeed(BinanceDataFeed):
            def __init__(self) -> None:
                super().__init__("BTCUSDT", "1d", 10)
                self._bars   = bars
                self._cursor = 0
            async def load(self) -> None:
                pass

        strategy = RecordingStrategy()
        engine   = BacktestEngine(
            feed=StaticFeed(),
            strategy=strategy,
            portfolio=portfolio,
            broker=broker,
            warm_up_bars=0,
        )

        # Act
        engine.run()

        # Assert — all 10 bars were processed in order
        assert strategy.seen_indices == list(range(10)), (
            f"Expected bars 0–9 in order, got {strategy.seen_indices}"
        )

    def test_engine_processes_correct_number_of_bars(
        self,
        broker:    SimulatedBroker,
        portfolio: Portfolio,
    ) -> None:
        """
        After run() completes, bar_count must equal the total number of bars
        in the feed. A lower count means the engine terminated early.
        """
        # Arrange
        from src.backtester.data.binance import BinanceDataFeed

        n_bars = 20
        bars   = build_static_feed(n_bars)

        class StaticFeed(BinanceDataFeed):
            def __init__(self) -> None:
                super().__init__("BTCUSDT", "1d", n_bars)
                self._bars   = bars
                self._cursor = 0
            async def load(self) -> None:
                pass

        strategy = RecordingStrategy()
        engine   = BacktestEngine(
            feed=StaticFeed(),
            strategy=strategy,
            portfolio=portfolio,
            broker=broker,
            warm_up_bars=0,
        )

        # Act
        engine.run()

        # Assert
        assert engine.bar_count == n_bars, (
            f"Expected {n_bars} bars processed, got {engine.bar_count}"
        )

    def test_signals_during_warmup_do_not_produce_orders(
        self,
        broker: SimulatedBroker,
    ) -> None:
        """
        Any signal emitted during the warm-up window must be discarded.
        The portfolio must have zero trades after a run where all bars
        fall within the warm-up period.
        """
        # Arrange
        from src.backtester.data.binance import BinanceDataFeed
        from src.backtester.strategies.ma_crossover import SMACrossoverStrategy

        # Build enough bars to trigger an SMA crossover (bar 50+)
        # but set warm_up_bars high enough to cover all of them.
        prices: list[float] = [100.0] * 50 + [200.0] * 10
        bars: list[MarketEvent] = [
            make_bar(i, p, p, p, p) for i, p in enumerate(prices)
        ]

        class StaticFeed(BinanceDataFeed):
            def __init__(self) -> None:
                super().__init__("BTCUSDT", "1d", len(bars))
                self._bars   = bars
                self._cursor = 0
            async def load(self) -> None:
                pass

        portfolio = Portfolio(initial_capital=10_000.0)
        strategy  = SMACrossoverStrategy(fast_period=20, slow_period=50)
        engine    = BacktestEngine(
            feed=StaticFeed(),
            strategy=strategy,
            portfolio=portfolio,
            broker=broker,
            warm_up_bars=999,  # warm-up covers all bars
        )

        # Act
        engine.run()

        # Assert — no trades should have been placed
        assert len(portfolio.trades) == 0, (
            f"Expected 0 trades during warm-up, got {len(portfolio.trades)}"
        )
