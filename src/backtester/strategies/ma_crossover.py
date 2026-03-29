"""
strategies/ma_crossover.py
--------------------------
SMA 20 / SMA 50 crossover strategy — the simplest possible trend-following
strategy. Used as the first strategy because every signal can be hand-
verified against a candlestick chart, which makes engine bugs obvious.

Signal logic
------------
- When SMA20 crosses ABOVE SMA50 → emit LONG signal (enter position)
- When SMA20 crosses BELOW SMA50 → emit EXIT signal (close position, go flat)
- All other bars                  → no signal (None)

The strategy is long-only for now. EXIT means go flat, not short.
This design is portable to futures: the signal direction enum supports
SHORT as a future extension without changing this file.

Warm-up
-------
The first valid SMA50 value requires 50 closes (bars 0–49).
A crossover requires two consecutive SMA50 values to compare prev vs current,
so the earliest possible signal is bar 50.
Bars before that are accumulated silently with no signal emitted.
"""

from src.backtester.data.events import MarketEvent, SignalEvent, SignalDirection
from src.backtester.strategies.base import Strategy


class SMACrossoverStrategy(Strategy):
    """
    SMA 20 / SMA 50 crossover. Long-only, bar-by-bar, zero lookahead.

    Parameters
    ----------
    fast_period: lookback for the fast SMA (default 20)
    slow_period: lookback for the slow SMA (default 50)
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50) -> None:
        self._fast_period: int = fast_period
        self._slow_period: int = slow_period

        # Rolling close price buffer — capped at slow_period to bound memory.
        self._closes: list[float] = []

        # Previous bar's relative position of fast vs slow SMA.
        # None = not yet enough data; True = fast was above slow; False = below.
        self._prev_fast_above_slow: bool | None = None

        # Whether we currently hold a position — prevents redundant signals.
        self._in_position: bool = False

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        """
        Ingest the latest close, recompute SMAs, and emit a signal if a
        crossover occurred since the previous bar.
        """
        # Update the rolling close buffer.
        self._closes.append(event.close)
        if len(self._closes) > self._slow_period:
            self._closes.pop(0)

        # Cannot compute SMA50 until we have 50 closes.
        if len(self._closes) < self._slow_period:
            return None

        # Compute both SMAs from the current buffer.
        fast_sma: float = sum(self._closes[-self._fast_period:]) / self._fast_period
        slow_sma: float = sum(self._closes) / self._slow_period

        current_fast_above_slow: bool = fast_sma > slow_sma

        # First bar with both SMAs computable — record state, no signal yet.
        # We need a previous state to detect a crossover (transition), so we
        # wait one more bar before emitting anything.
        if self._prev_fast_above_slow is None:
            self._prev_fast_above_slow = current_fast_above_slow
            return None

        signal: SignalEvent | None = None

        # Golden cross: fast crosses above slow → enter long.
        if current_fast_above_slow and not self._prev_fast_above_slow:
            if not self._in_position:
                signal = SignalEvent(
                    symbol=event.symbol,
                    timestamp=event.timestamp,
                    direction=SignalDirection.LONG,
                )
                self._in_position = True

        # Death cross: fast crosses below slow → exit long, go flat.
        elif not current_fast_above_slow and self._prev_fast_above_slow:
            if self._in_position:
                signal = SignalEvent(
                    symbol=event.symbol,
                    timestamp=event.timestamp,
                    direction=SignalDirection.EXIT,
                )
                self._in_position = False

        # Advance crossover state for the next bar.
        self._prev_fast_above_slow = current_fast_above_slow

        return signal

    def reset(self) -> None:
        """Clear all internal state for walk-forward window reuse."""
        self._closes.clear()
        self._prev_fast_above_slow = None
        self._in_position = False
