"""
test_fill_timing.py
-------------------
Regression tests for the fill timing of the engine.

A signal on bar N must fill at the open of bar N+1, with the timestamp of
bar N+1. The equity of a bar must not include a fill from a later bar.

Each bar has open = 1000 + bar_index and close = open, and the broker has
zero slippage. Thus the fill price shows the bar of the fill directly
(broker.py: fill_price = next_bar.open * (1 + 0.0)).
"""

from src.backtester.broker import SimulatedBroker
from src.backtester.data.base import DataFeed
from src.backtester.data.events import (
    FillEvent,
    MarketEvent,
    OrderSide,
    SignalDirection,
    SignalEvent,
)
from src.backtester.engine import BacktestEngine
from src.backtester.portfolio import Portfolio
from src.backtester.strategies.base import Strategy
from tests.helpers import make_bar

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ListFeed(DataFeed):
    """A DataFeed that serves a fixed list of bars. It has no network code."""

    def __init__(self, bars: list[MarketEvent]) -> None:
        self._bars: list[MarketEvent] = bars
        self._cursor: int = 0

    async def load(self) -> None:
        pass  # The bars are already in memory.

    def next_bar(self) -> MarketEvent | None:
        if self._cursor >= len(self._bars):
            return None
        bar: MarketEvent = self._bars[self._cursor]
        self._cursor += 1
        return bar

    def __len__(self) -> int:
        return len(self._bars)

    def get_bar(self, index: int) -> MarketEvent:
        return self._bars[index]

    def reset(self) -> None:
        self._cursor = 0


class ScriptedStrategy(Strategy):
    """Emits a fixed signal on each bar index in the script, else nothing."""

    def __init__(self, script: dict[int, SignalDirection]) -> None:
        self._script: dict[int, SignalDirection] = script

    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        direction: SignalDirection | None = self._script.get(event.bar_index)
        if direction is None:
            return None
        return SignalEvent(
            symbol=event.symbol,
            timestamp=event.timestamp,
            direction=direction,
        )

    def reset(self) -> None:
        pass  # The script holds no state that changes.


class RecordingPortfolio(Portfolio):
    """
    A Portfolio that records each fill, and the length of the equity curve
    at the time of the fill. The next equity point after the fill is the
    first equity point that includes the fill.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fills: list[FillEvent] = []
        self.equity_len_at_fill: list[int] = []

    def on_fill(self, fill: FillEvent) -> None:
        super().on_fill(fill)
        self.fills.append(fill)
        self.equity_len_at_fill.append(len(self.equity_curve))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_bars(n_bars: int) -> list[MarketEvent]:
    """Build bars with open = close = 1000 + bar_index."""
    return [
        make_bar(
            i,
            open_=1000.0 + i,
            high=1005.0 + i,
            low=995.0 + i,
            close=1000.0 + i,
        )
        for i in range(n_bars)
    ]


def run_engine(
    bars: list[MarketEvent],
    script: dict[int, SignalDirection],
) -> RecordingPortfolio:
    """Run the engine with new components and no warm-up."""
    portfolio = RecordingPortfolio()
    engine = BacktestEngine(
        feed=ListFeed(bars),
        strategy=ScriptedStrategy(script),
        portfolio=portfolio,
        broker=SimulatedBroker(fee_rate=0.001, slippage=0.0),
        warm_up_bars=0,
    )
    engine.run()
    return portfolio


def fill_key(fill: FillEvent) -> tuple[int, OrderSide, float, float, float]:
    """The fill fields that two runs must agree on."""
    return (
        fill.timestamp,
        fill.side,
        fill.quantity,
        fill.fill_price,
        fill.commission,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFillTiming:

    def test_long_signal_on_bar_5_fills_at_open_of_bar_6(self) -> None:
        # Arrange
        bars = build_bars(10)

        # Act
        portfolio = run_engine(bars, {5: SignalDirection.LONG})

        # Assert
        assert len(portfolio.fills) == 1
        fill = portfolio.fills[0]
        assert fill.side == OrderSide.BUY
        assert fill.fill_price == bars[6].open, (
            f"Expected fill at open of bar 6 ({bars[6].open}), got {fill.fill_price}"
        )
        assert fill.timestamp == bars[6].timestamp, (
            f"Expected timestamp of bar 6 ({bars[6].timestamp}), got {fill.timestamp}"
        )

    def test_fill_timestamp_equals_first_equity_point_that_includes_the_fill(
        self,
    ) -> None:
        # Arrange
        bars = build_bars(12)
        script = {5: SignalDirection.LONG, 8: SignalDirection.EXIT}

        # Act
        portfolio = run_engine(bars, script)

        # Assert
        assert len(portfolio.fills) == 2
        for fill, equity_len in zip(
            portfolio.fills, portfolio.equity_len_at_fill, strict=True
        ):
            equity_ts: int = portfolio.equity_curve[equity_len][0]
            assert equity_ts == fill.timestamp, (
                f"Fill at {fill.timestamp} first appears in the equity of the bar "
                f"at {equity_ts}"
            )

    def test_signal_on_last_bar_gives_no_fill(self) -> None:
        # Arrange
        bars = build_bars(10)

        # Act
        portfolio = run_engine(bars, {9: SignalDirection.LONG})

        # Assert
        assert portfolio.fills == []
        assert portfolio.trades == []

    def test_exit_signal_on_bar_8_fills_at_open_of_bar_9(self) -> None:
        # Arrange
        bars = build_bars(12)
        script = {5: SignalDirection.LONG, 8: SignalDirection.EXIT}

        # Act
        portfolio = run_engine(bars, script)

        # Assert
        assert len(portfolio.fills) == 2
        exit_fill = portfolio.fills[1]
        assert exit_fill.side == OrderSide.SELL
        assert exit_fill.fill_price == bars[9].open, (
            f"Expected fill at open of bar 9 ({bars[9].open}), "
            f"got {exit_fill.fill_price}"
        )
        assert exit_fill.timestamp == bars[9].timestamp
        assert len(portfolio.trades) == 1
        trade = portfolio.trades[0]
        assert trade.entry_time == bars[6].timestamp
        assert trade.exit_time == bars[9].timestamp

    def test_prefix_run_matches_full_run(self) -> None:
        """
        A run on the first n bars must give the same equity curve and the
        same fills as the first n bars of a run on all bars. If the engine
        uses a later bar, the two runs differ.
        """
        # Arrange
        bars = build_bars(20)
        script = {
            5: SignalDirection.LONG,
            8: SignalDirection.EXIT,
            12: SignalDirection.LONG,
            15: SignalDirection.EXIT,
        }
        full = run_engine(bars, script)

        for n in range(1, len(bars) + 1):
            # Act
            prefix = run_engine(bars[:n], script)

            # Assert
            assert prefix.equity_curve == full.equity_curve[:n], (
                f"Equity curve differs for the prefix of {n} bars"
            )
            last_ts: int = bars[n - 1].timestamp
            expected_fills = [
                fill_key(f) for f in full.fills if f.timestamp <= last_ts
            ]
            assert [fill_key(f) for f in prefix.fills] == expected_fills, (
                f"Fills differ for the prefix of {n} bars"
            )
