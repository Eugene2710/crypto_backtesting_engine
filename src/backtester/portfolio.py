"""
portfolio.py
------------
Portfolio: tracks positions, cash, equity curve, and ATR-based position sizing.

Responsibilities
----------------
1. Receive SignalEvents from the strategy and size them into OrderEvents
   using ATR-based position sizing (risk a fixed % of equity per trade).
2. Receive FillEvents from the broker and update cash, position, and
   the running equity curve.
3. Record every trade (entry + exit pair) for performance analysis.

ATR-based position sizing
--------------------------
Rather than always buying a fixed dollar amount, we size each position so
that a 1-ATR adverse move costs exactly `risk_per_trade` fraction of current
equity. This means:

    quantity = (equity × risk_per_trade) / ATR

Example: equity=$10,000, risk=1%, ATR=$2,000
    quantity = (10,000 × 0.01) / 2,000 = 0.05 BTC

If a 1-ATR adverse move occurs:
    loss = 0.05 × $2,000 = $100 = exactly 1% of equity ✓

If ATR is not yet available (fewer than atr_period bars seen), the position
is sized using a fallback: equity × risk_per_trade / price — equivalent
to a fixed 1% equity bet with no volatility adjustment.

ATR Calculation -  a measure of how much the price typically moves in a single bar.

True Range =  max(
    high-low, # full intra bar range
    |high - previous close|, # inter bar range
    |low - previous close|) # inter bar range
)

Long-only
---------
EXIT signals close the entire open position (sell all held quantity).
There is no shorting in this implementation. Short support is a planned
future extension — the OrderSide enum already supports it.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.backtester.events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    SignalDirection,
    SignalEvent,
)


# ---------------------------------------------------------------------------
# Trade record — one completed round trip (entry fill + exit fill)
# ---------------------------------------------------------------------------

class Trade(BaseModel):
    """Records a completed long trade (entry → exit)."""
    symbol:       str
    entry_time:   int    # timestamp of the entry fill bar (ms)
    exit_time:    int    # timestamp of the exit fill bar  (ms)
    entry_price:  float  # actual fill price at entry (after slippage)
    exit_price:   float  # actual fill price at exit  (after slippage)
    quantity:     float
    gross_pnl:    float  # (exit_price - entry_price) × quantity
    total_fees:   float  # entry commission + exit commission
    net_pnl:      float  # gross_pnl - total_fees


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class Portfolio:
    """
    Tracks cash, position, equity curve, and completed trades.

    Parameters
    ----------
    initial_capital : starting cash in quote currency (e.g. USDT)
    risk_per_trade  : fraction of equity to risk per trade (default 1% = 0.01)
                      Tune this after reviewing Sharpe / drawdown from backtests.
    atr_period      : lookback for ATR calculation (default 14 bars)
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_per_trade:  float = 0.01,
        atr_period:      int   = 14,
    ) -> None:
        self._initial_capital: float = initial_capital
        self._risk_per_trade:  float = risk_per_trade
        self._atr_period:      int   = atr_period

        self.cash: float = initial_capital          # quote currency balance (e.g. USDT)
        self.position_qty: float = 0.0              # base asset held (e.g. BTC); 0 = flat

        # Fill price at which the current position was entered.
        # Carried forward to compute gross P&L at exit.
        self._entry_price: float = 0.0

        # Commission paid at entry — added to exit commission when recording
        # the completed trade so total_fees reflects the full round trip.
        self._entry_commission: float = 0.0

        # Timestamp of the entry fill — carried forward for the trade record.
        self._entry_time: int = 0

        # Previous bar's close — required to compute True Range on the next bar.
        # None on the very first bar (no prior close exists).
        self._prev_close: float | None = None

        # Rolling True Range buffer for ATR calculation.
        # Capped at atr_period entries to keep memory bounded.
        self._true_ranges: list[float] = []

        # Equity curve: (timestamp_ms, equity) recorded after every bar.
        # Equity = cash + mark-to-market value of open position at bar close.
        # Public — read directly, treat as read-only.
        self.equity_curve: list[tuple[int, float]] = []

        # All completed round-trip trades, used by performance.py.
        # Public — read directly, treat as read-only.
        self.trades: list[Trade] = []

    # ------------------------------------------------------------------
    # Public interface — called by the engine in this order each bar:
    #   1. on_signal  (if strategy emits a signal)
    #   2. on_fill    (if broker confirms a fill)
    #   3. on_bar     (always — updates ATR and equity curve)
    # ------------------------------------------------------------------

    def on_signal(self, signal: SignalEvent, current_bar: MarketEvent) -> OrderEvent | None:
        """
        Convert a SignalEvent into a sized OrderEvent.

        Parameters
        ----------
        signal      : SignalEvent emitted by the strategy on bar N
        current_bar : the bar on which the signal fired (bar N)
                      Price used for position sizing and affordability check.

        Returns
        -------
        OrderEvent if an order should be placed, else None.
        Returns None on redundant signals (LONG while already long, EXIT while flat).
        """
        if signal.direction == SignalDirection.LONG:
            # Guard: do not double-enter an already open position.
            if self.position_qty > 0:
                return None

            quantity: float = self._compute_position_size(current_bar)
            if quantity <= 0:
                return None

            return OrderEvent(
                symbol=signal.symbol,
                timestamp=signal.timestamp,
                side=OrderSide.BUY,
                quantity=quantity,
            )

        elif signal.direction == SignalDirection.EXIT:
            # Guard: nothing to exit if already flat.
            if self.position_qty <= 0:
                return None

            # Always exit the full position — no partial closes in this version.
            return OrderEvent(
                symbol=signal.symbol,
                timestamp=signal.timestamp,
                side=OrderSide.SELL,
                quantity=self.position_qty,
            )

        return None

    def on_fill(self, fill: FillEvent) -> None:
        """
        Update cash, position state, and trade records on a confirmed fill.

        Parameters
        ----------
        fill : FillEvent emitted by the SimulatedBroker for bar N+1
        """
        if fill.side == OrderSide.BUY:
            # Deduct total purchase cost (price × qty + commission) from cash.
            total_cost: float = fill.fill_price * fill.quantity + fill.commission
            self.cash -= total_cost

            # Store entry details for use when the position is eventually closed.
            self.position_qty     = fill.quantity
            self._entry_price      = fill.fill_price
            self._entry_commission = fill.commission
            self._entry_time       = fill.timestamp

        elif fill.side == OrderSide.SELL:
            # Add net sale proceeds (price × qty − commission) to cash.
            total_proceeds: float = fill.fill_price * fill.quantity - fill.commission
            self.cash += total_proceeds

            # Compute P&L for the completed round trip.
            gross_pnl:  float = (fill.fill_price - self._entry_price) * fill.quantity
            total_fees: float = self._entry_commission + fill.commission

            self.trades.append(Trade(
                symbol=fill.symbol,
                entry_time=self._entry_time,
                exit_time=fill.timestamp,
                entry_price=self._entry_price,
                exit_price=fill.fill_price,
                quantity=fill.quantity,
                gross_pnl=gross_pnl,
                total_fees=total_fees,
                net_pnl=gross_pnl - total_fees,
            ))

            # Clear position state — we are now flat.
            self.position_qty     = 0.0
            self._entry_price      = 0.0
            self._entry_commission = 0.0
            self._entry_time       = 0

    def on_bar(self, bar: MarketEvent) -> None:
        """
        Update ATR buffer and snapshot the current equity.

        Called by the engine on every bar AFTER fills are processed so
        that the equity curve reflects any position changes on this bar.

        Parameters
        ----------
        bar : the current MarketEvent (bar N)
        """
        # Update rolling True Range buffer for next position size calculation.
        self._update_atr(bar)

        # Mark-to-market: value the open position at this bar's close price.
        mark_to_market: float = self.position_qty * bar.close
        equity: float = self.cash + mark_to_market
        self.equity_curve.append((bar.timestamp, equity))

        # Store close for True Range calculation on the next bar.
        self._prev_close = bar.close

    @property
    def equity(self) -> float:
        """Current equity: cash + mark-to-market. Returns initial capital before any bars."""
        return self.equity_curve[-1][1] if self.equity_curve else self._initial_capital

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_position_size(self, bar: MarketEvent) -> float:
        """
        ATR-based position sizing.

        Size the position so that a 1-ATR adverse move equals
        exactly risk_per_trade × current_equity.

            quantity = (equity × risk_per_trade) / ATR

        Falls back to (equity × risk_per_trade) / price when ATR
        is not yet available.

        Always capped at what cash can afford at the current bar's close.
        """
        risk_amount: float  = self.equity * self._risk_per_trade
        atr: float | None   = self._current_atr()

        if atr is not None and atr > 0:
            quantity: float = risk_amount / atr
        else:
            # Fallback: no volatility adjustment, size purely by equity fraction.
            quantity = risk_amount / bar.close if bar.close > 0 else 0.0

        # Cap at the maximum quantity we can actually afford with available cash.
        max_affordable: float = self.cash / bar.close if bar.close > 0 else 0.0
        return min(quantity, max_affordable)

    def _update_atr(self, bar: MarketEvent) -> None:
        """
        Compute True Range for the current bar and append to the buffer.

        True Range = max of:
          - high − low                  (intrabar range)
          - |high − previous close|     (overnight gap up)
          - |low  − previous close|     (overnight gap down)

        On the very first bar there is no previous close, so True Range
        defaults to high − low.
        """
        if self._prev_close is None:
            true_range: float = bar.high - bar.low
        else:
            true_range = max(
                bar.high - bar.low,
                abs(bar.high - self._prev_close),
                abs(bar.low  - self._prev_close),
            )

        self._true_ranges.append(true_range)

        # Keep buffer bounded — only the last atr_period values are needed.
        if len(self._true_ranges) > self._atr_period:
            self._true_ranges.pop(0)

    def _current_atr(self) -> float | None:
        """
        Return the simple average of the True Range buffer, or None if
        fewer than atr_period bars have been processed.
        """
        if len(self._true_ranges) < self._atr_period:
            return None
        return sum(self._true_ranges) / self._atr_period
