"""
test_signals.py
---------------
Phase 3 unit tests — SMA crossover signal correctness.

These tests verify that SMACrossoverStrategy emits signals at exactly
the right bars on a hand-crafted price series whose crossover points are
known in advance. Any engine bug that causes a signal to fire on the wrong
bar will be immediately visible here.

Price series design
-------------------
  Bars   0–49 : close = 100  (flat — SMA20 == SMA50, no crossover possible)
  Bars  50–54 : close = 200  (spike up — SMA20 rises above SMA50 → golden cross)
  Bars  55–104: close = 50   (crash down — SMA20 falls below SMA50 → death cross)

Expected behaviour
------------------
  - No signal during bars 0–49 (warm-up, SMAs equal)
  - Exactly one LONG signal during bars 50–54 (golden cross)
  - Exactly one EXIT signal during bars 55–104 (death cross)
  - No signal before bar 50 (insufficient history for crossover detection)

These are pure unit tests: no network calls, no broker, no portfolio.
Only the strategy and synthetic MarketEvents are involved.

Principles applied
------------------
- Each test verifies exactly one signal property (count, direction, timing).
- The price series is designed so crossover points are mathematically
  guaranteed — there is no ambiguity about when signals should fire.
- Fixtures provide a fresh strategy instance for each test.
"""

import pytest

from conftest import make_bar
from src.backtester.events import MarketEvent, SignalEvent, SignalDirection
from src.backtester.strategies.ma_crossover import SMACrossoverStrategy


# ---------------------------------------------------------------------------
# Shared price series construction
# ---------------------------------------------------------------------------

def build_crossover_price_series() -> list[float]:
    """
    Construct a price series with guaranteed SMA 20/50 crossover points.

      Bars   0–49 : 100 (flat baseline)
      Bars  50–54 : 200 (golden cross region)
      Bars  55–104: 50  (death cross region)
    """
    return [100.0] * 50 + [200.0] * 5 + [50.0] * 50


def run_strategy(
    strategy: SMACrossoverStrategy,
    prices: list[float],
) -> list[SignalEvent]:
    """Feed all prices through the strategy and collect every signal emitted."""
    signals: list[SignalEvent] = []
    for i, price in enumerate(prices):
        bar: MarketEvent = make_bar(i, open_=price, high=price, low=price, close=price)
        signal: SignalEvent | None = strategy.on_bar(bar)
        if signal is not None:
            signals.append(signal)
    return signals


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSMACrossoverSignals:

    def test_no_signal_emitted_during_flat_baseline_period(
        self, sma_strategy: SMACrossoverStrategy
    ) -> None:
        """
        During bars 0–49 where all closes are equal, SMA20 == SMA50 at all
        times. No crossover can occur so no signal should ever be emitted.
        """
        # Arrange
        flat_prices: list[float] = [100.0] * 50

        # Act
        signals: list[SignalEvent] = run_strategy(sma_strategy, flat_prices)

        # Assert
        assert len(signals) == 0, (
            f"Expected no signals during flat baseline, got {len(signals)} signals."
        )

    def test_exactly_one_long_signal_emitted_on_golden_cross(
        self, sma_strategy: SMACrossoverStrategy
    ) -> None:
        """
        When price spikes from 100 to 200 after bar 50, SMA20 crosses above
        SMA50. This must produce exactly one LONG signal — not zero (missed)
        and not more than one (duplicate).
        """
        # Arrange
        prices = build_crossover_price_series()

        # Act
        signals = run_strategy(sma_strategy, prices)
        long_signals = [s for s in signals if s.direction == SignalDirection.LONG]

        # Assert
        assert len(long_signals) == 1, (
            f"Expected exactly 1 LONG signal, got {len(long_signals)}."
        )

    def test_golden_cross_fires_within_spike_window(
        self, sma_strategy: SMACrossoverStrategy
    ) -> None:
        """
        The LONG signal must fire within bars 50–54 (the spike-up window).
        Firing before bar 50 would be lookahead; firing after bar 54 would
        mean the strategy missed the crossover entirely.
        """
        # Arrange
        prices = build_crossover_price_series()
        bar_indices: dict[int, SignalEvent] = {}

        # Act — track which bar each signal fires on
        for i, price in enumerate(prices):
            bar = make_bar(i, open_=price, high=price, low=price, close=price)
            signal = sma_strategy.on_bar(bar)
            if signal is not None and signal.direction == SignalDirection.LONG:
                bar_indices[i] = signal

        # Assert
        assert len(bar_indices) == 1
        signal_bar: int = list(bar_indices.keys())[0]
        assert 50 <= signal_bar <= 54, (
            f"Golden cross fired at bar {signal_bar}, expected between bars 50–54."
        )

    def test_exactly_one_exit_signal_emitted_on_death_cross(
        self, sma_strategy: SMACrossoverStrategy
    ) -> None:
        """
        When price crashes from 200 to 50 after bar 54, SMA20 crosses below
        SMA50. This must produce exactly one EXIT signal.
        """
        # Arrange
        prices = build_crossover_price_series()

        # Act
        signals = run_strategy(sma_strategy, prices)
        exit_signals = [s for s in signals if s.direction == SignalDirection.EXIT]

        # Assert
        assert len(exit_signals) == 1, (
            f"Expected exactly 1 EXIT signal, got {len(exit_signals)}."
        )

    def test_death_cross_fires_after_spike_window(
        self, sma_strategy: SMACrossoverStrategy
    ) -> None:
        """
        The EXIT signal must fire at bar 55 or later — the crash window.
        Firing during the spike window (bars 50–54) would be impossible
        given the price construction and would indicate a logic error.
        """
        # Arrange
        prices = build_crossover_price_series()
        bar_indices: dict[int, SignalEvent] = {}

        # Act
        for i, price in enumerate(prices):
            bar = make_bar(i, open_=price, high=price, low=price, close=price)
            signal = sma_strategy.on_bar(bar)
            if signal is not None and signal.direction == SignalDirection.EXIT:
                bar_indices[i] = signal

        # Assert
        assert len(bar_indices) == 1
        signal_bar = list(bar_indices.keys())[0]
        assert signal_bar >= 55, (
            f"Death cross fired at bar {signal_bar}, expected at bar ≥ 55."
        )

    def test_total_signal_count_is_exactly_two(
        self, sma_strategy: SMACrossoverStrategy
    ) -> None:
        """
        Over the full price series there must be exactly two signals:
        one LONG (golden cross) and one EXIT (death cross). Any additional
        signals would indicate spurious crossover detection.
        """
        # Arrange
        prices = build_crossover_price_series()

        # Act
        signals = run_strategy(sma_strategy, prices)

        # Assert
        assert len(signals) == 2, (
            f"Expected exactly 2 signals total, got {len(signals)}: "
            f"{[s.direction for s in signals]}"
        )

    def test_reset_clears_all_internal_state(
        self, sma_strategy: SMACrossoverStrategy
    ) -> None:
        """
        After reset(), the strategy must behave identically to a freshly
        constructed instance. This is required for walk-forward testing
        where the same strategy object is reused across windows.

        Verified by running the full price series twice — once before reset
        and once after — and asserting both runs produce the same signals
        at the same bar indices.
        """
        # Arrange
        prices = build_crossover_price_series()

        # Act — first run
        signals_first: list[tuple[int, SignalDirection]] = []
        for i, price in enumerate(prices):
            bar = make_bar(i, open_=price, high=price, low=price, close=price)
            signal = sma_strategy.on_bar(bar)
            if signal is not None:
                signals_first.append((i, signal.direction))

        sma_strategy.reset()

        # Act — second run after reset
        signals_second: list[tuple[int, SignalDirection]] = []
        for i, price in enumerate(prices):
            bar = make_bar(i, open_=price, high=price, low=price, close=price)
            signal = sma_strategy.on_bar(bar)
            if signal is not None:
                signals_second.append((i, signal.direction))

        # Assert
        assert signals_first == signals_second, (
            f"Signals after reset differ from first run.\n"
            f"First:  {signals_first}\n"
            f"Second: {signals_second}"
        )
