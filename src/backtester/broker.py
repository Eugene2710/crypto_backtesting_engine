"""
broker.py
---------
SimulatedBroker: converts OrderEvents into FillEvents.

Fill model
----------
- Fill price : next bar's open (conservative, realistic assumption —
               you cannot fill at the signal bar's close in practice)
- Commission : configurable taker fee applied to both entry and exit.
               Default 0.1% per side (0.2% round trip), matching Binance
               spot taker fee.
- Slippage   : configurable flat basis-point deduction from fill price.
               Default 0 to start; positive slippage worsens the fill
               (buy higher, sell lower).

Why next-bar open?
------------------
A signal fires at the CLOSE of bar N. In live trading you cannot execute
at that close price — the market has already moved. The earliest realistic
fill is the OPEN of bar N+1, which is what this broker uses. This is the
most conservative (pessimistic) fill assumption and avoids lookahead bias.
"""

from src.backtester.data.events import FillEvent, MarketEvent, OrderEvent, OrderSide


class SimulatedBroker:
    """
    Simulates order execution at the next bar's open with fees and slippage.

    Parameters
    ----------
    fee_rate : taker fee as a decimal fraction (default 0.001 = 0.1%)
    slippage : slippage in decimal fraction of fill price (default 0.0)
               Applied adversely: buys fill higher, sells fill lower.
    """

    def __init__(
        self,
        fee_rate: float = 0.001,
        slippage: float = 0.0,
    ) -> None:
        self._fee_rate: float = fee_rate
        self._slippage: float = slippage

    # ------------------------------------------------------------------
    # Core execution method
    # ------------------------------------------------------------------

    def execute(self, order: OrderEvent, next_bar: MarketEvent) -> FillEvent:
        """
        Fill an order at the next bar's open price with fees and slippage.

        Parameters
        ----------
        order    : the OrderEvent raised by the portfolio on bar N
        next_bar : the MarketEvent for bar N+1 (provides the fill price)

        Returns
        -------
        FillEvent with all cost components populated.
        """
        # Base fill price is next bar's open — no lookahead beyond this.
        raw_fill_price: float = next_bar.open

        # Apply slippage adversely:
        #   BUY  → fill price is slightly higher (we pay more)
        #   SELL → fill price is slightly lower  (we receive less)
        if order.side == OrderSide.BUY:
            fill_price: float = raw_fill_price * (1 + self._slippage)
        else:
            fill_price = raw_fill_price * (1 - self._slippage)

        # Commission is calculated on the actual fill value after slippage.
        # This matches how real exchanges calculate taker fees.
        fill_value: float  = fill_price * order.quantity
        commission: float  = fill_value * self._fee_rate
        slippage_cost: float = abs(fill_price - raw_fill_price) * order.quantity

        return FillEvent(
            symbol=order.symbol,
            timestamp=next_bar.timestamp,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_cost,
        )
