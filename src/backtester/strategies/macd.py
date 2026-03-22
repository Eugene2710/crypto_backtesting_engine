"""
strategies/macd.py
------------------
MACD (12, 26, 9) crossover strategy.

Signal logic
------------
- When MACD line crosses ABOVE the signal line → emit LONG  (enter position)
- When MACD line crosses BELOW the signal line → emit EXIT  (close position, go flat)

MACD internals
--------------
  MACD line   = EMA(12) − EMA(26)
  Signal line = EMA(9) of the MACD line
  Histogram   = MACD line − Signal line  (computed but not used for signals here)

Warm-up sequence
----------------
  Bar 0–25:  accumulating closes to seed EMA(26) — no output
  Bar 26:    first EMA(12) and EMA(26) values → first MACD value
  Bar 27–33: accumulating MACD values to seed EMA(9) signal line
  Bar 34:    first signal line value
  Bar 35+:   crossovers detectable (need two consecutive signal line values)

EMA implementation
------------------
Computed incrementally bar-by-bar using standard smoothing factor k = 2/(period+1),
seeded with the SMA of the first `period` values. This matches crypto_indicators'
calc_ema exactly, so MACD values here will agree with calc_macd_series output.

Why incremental instead of calling calc_macd_series?
  calc_macd_series rescans the full close history on every bar — O(n²) total.
  Incremental EMA is O(1) per bar — O(n) total. For 500 bars this doesn't
  matter, but the habit is correct for future higher-frequency data (4H, 1H).
"""

from __future__ import annotations

from src.backtester.events import MarketEvent, SignalEvent, SignalDirection
from src.backtester.strategies.base import Strategy


class MACDCrossoverStrategy(Strategy):
    """
    MACD(12, 26, 9) crossover. Long-only, bar-by-bar, zero lookahead.

    Parameters
    ----------
    fast_period:   EMA period for the fast line (default 12)
    slow_period:   EMA period for the slow line (default 26)
    signal_period: EMA period for the signal line (default 9)
    """

    def __init__(
        self,
        fast_period:   int = 12,
        slow_period:   int = 26,
        signal_period: int = 9,
    ) -> None:
        self._fast_period:   int = fast_period
        self._slow_period:   int = slow_period
        self._signal_period: int = signal_period

        # Smoothing multipliers: k = 2 / (period + 1)
        self._k_fast:   float = 2.0 / (fast_period + 1)
        self._k_slow:   float = 2.0 / (slow_period + 1)
        self._k_signal: float = 2.0 / (signal_period + 1)

        # Phase 1 buffer: raw closes accumulated until EMA(slow) can be seeded.
        self._seed_closes: list[float] = []

        # EMA state — None until the seed SMA has been computed.
        self._ema_fast:   float | None = None
        self._ema_slow:   float | None = None

        # Phase 3 buffer: MACD values accumulated until signal EMA can be seeded.
        self._macd_seed_buf: list[float] = []

        # Signal line EMA — None until signal_period MACD values are accumulated.
        self._ema_signal: float | None = None

        # Previous bar's MACD-vs-signal relationship for crossover detection.
        # None = not yet enough data; True = MACD above signal; False = below.
        self._prev_macd_above_signal: bool | None = None

        # Position tracking — prevents redundant LONG signals while already long.
        self._in_position: bool = False

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        """
        Incrementally update all three EMAs and emit a signal on crossover.
        """
        close: float = event.close

        # ---- Phase 1: accumulate closes until EMA(slow) can be seeded -------
        # We need slow_period closes to compute the seed SMA for EMA(slow).
        # EMA(fast) is seeded from the last fast_period values of the same window.
        if self._ema_slow is None:
            self._seed_closes.append(close)

            if len(self._seed_closes) == self._slow_period:
                # Seed EMA(fast) and EMA(slow) simultaneously from the same window.
                # EMA(fast) seeds from the last fast_period closes in the window
                # so both EMAs are aligned to the same bar.
                self._ema_fast = (
                    sum(self._seed_closes[-self._fast_period:]) / self._fast_period
                )
                self._ema_slow = sum(self._seed_closes) / self._slow_period

            return None

        # ---- Phase 2: update EMA(fast) and EMA(slow) each bar ---------------
        self._ema_fast = close * self._k_fast + self._ema_fast * (1 - self._k_fast)  # type: ignore[operator]
        self._ema_slow = close * self._k_slow + self._ema_slow * (1 - self._k_slow)

        macd_val: float = self._ema_fast - self._ema_slow

        # ---- Phase 3: accumulate MACD values to seed the signal line EMA ----
        if self._ema_signal is None:
            self._macd_seed_buf.append(macd_val)

            if len(self._macd_seed_buf) == self._signal_period:
                # Seed signal line EMA with the SMA of the first signal_period
                # MACD values.
                self._ema_signal = sum(self._macd_seed_buf) / self._signal_period

            return None

        # ---- Phase 4: update signal EMA and check for crossover --------------
        self._ema_signal = (
            macd_val * self._k_signal + self._ema_signal * (1 - self._k_signal)
        )

        current_macd_above_signal: bool = macd_val > self._ema_signal

        # First bar with a valid signal line — record state, no signal yet.
        # Need a previous state to detect a transition (crossover).
        if self._prev_macd_above_signal is None:
            self._prev_macd_above_signal = current_macd_above_signal
            return None

        signal: SignalEvent | None = None

        # Bullish crossover: MACD crosses above signal line → enter long.
        if current_macd_above_signal and not self._prev_macd_above_signal:
            if not self._in_position:
                signal = SignalEvent(
                    symbol=event.symbol,
                    timestamp=event.timestamp,
                    direction=SignalDirection.LONG,
                )
                self._in_position = True

        # Bearish crossover: MACD crosses below signal line → exit long, go flat.
        elif not current_macd_above_signal and self._prev_macd_above_signal:
            if self._in_position:
                signal = SignalEvent(
                    symbol=event.symbol,
                    timestamp=event.timestamp,
                    direction=SignalDirection.EXIT,
                )
                self._in_position = False

        # Advance crossover state for the next bar.
        self._prev_macd_above_signal = current_macd_above_signal

        return signal

    def reset(self) -> None:
        """Clear all internal state for walk-forward window reuse."""
        self._seed_closes.clear()
        self._macd_seed_buf.clear()
        self._ema_fast   = None
        self._ema_slow   = None
        self._ema_signal = None
        self._prev_macd_above_signal = None
        self._in_position = False
