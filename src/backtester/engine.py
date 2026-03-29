"""
engine.py
---------
Main event loop: feeds bars one at a time through the full event chain.

Event flow per bar
------------------
  Bar N close  →  Strategy.on_bar()      →  SignalEvent | None
  SignalEvent  →  Portfolio.on_signal()  →  OrderEvent  | None
  OrderEvent   →  held as pending order
  Bar N+1 open →  Broker.execute()       →  FillEvent
  FillEvent    →  Portfolio.on_fill()
  Always       →  Portfolio.on_bar()     →  ATR + equity curve updated

Zero lookahead guarantee
------------------------
The engine maintains a one-bar buffer: it processes bar N using bar N+1
only as the fill price source. The strategy and portfolio never receive
bar N+1 during bar N's processing. The pending order is stored between
iterations and filled at the very start of the next iteration, before
the strategy sees the new bar.

Warm-up period
--------------
The first `warm_up_bars` bars are fed to the strategy so indicator buffers
(SMA, EMA) have time to converge. Signals emitted during warm-up are
silently discarded — no orders are placed. Portfolio.on_bar() is still
called every bar so the ATR buffer is also fully warmed up by the time
the first trade fires.

Recommended warm_up_bars:
  ≥ 50  for SMA50-based strategies (minimum for first valid SMA)
  ≥ 200 for EMA-based strategies   (reduces seed bias to ~0%)
"""
from src.backtester.broker import SimulatedBroker
from src.backtester.data.base import DataFeed
from src.backtester.data.events import FillEvent, OrderEvent, SignalEvent
from src.backtester.portfolio import Portfolio
from src.backtester.strategies.base import Strategy
from src.backtester.data.events import MarketEvent


class BacktestEngine:
    """
    Orchestrates the bar-by-bar event loop across all components.

    Parameters
    ----------
    feed         : DataFeed already loaded via await feed.load()
    strategy     : Strategy instance (freshly constructed or reset)
    portfolio    : Portfolio instance
    broker       : SimulatedBroker instance
    warm_up_bars : bars fed to strategy without triggering any orders
    """

    def __init__(
        self,
        feed:         DataFeed,
        strategy:     Strategy,
        portfolio:    Portfolio,
        broker:       SimulatedBroker,
        warm_up_bars: int = 200,
    ) -> None:
        self._feed: DataFeed = feed
        self._strategy: Strategy = strategy
        self._portfolio: Portfolio = portfolio
        self._broker: SimulatedBroker = broker
        self._warm_up_bars: int = warm_up_bars

        # Pending order raised on bar N, to be filled at bar N+1's open.
        # Only one order can be pending at a time (one position at a time).
        self._pending_order: OrderEvent | None = None

        # Total bars processed across the full run (including warm-up).
        self.bar_count: int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full backtest by iterating over all bars in the feed.

        The feed must already be loaded before calling this method.
        Call feed.reset() before calling run() if replaying the same feed
        (e.g. for a second walk-forward window).
        """
        # Seed the one-bar buffer with the first bar from the feed.
        # We need to hold `prev_bar` so that on each iteration we can
        # process prev_bar while using current_bar as the fill source.

        prev_bar: MarketEvent | None = self._feed.next_bar()

        if prev_bar is None:
            return  # Empty feed — nothing to do.

        while True:
            current_bar: MarketEvent | None = self._feed.next_bar()

            # ---- Step 1: fill any pending order at current bar's open ----
            # This happens BEFORE the strategy sees current_bar, ensuring
            # the fill price is not accessible to strategy logic on prev_bar.
            # If there is no next bar (end of feed), the pending order is
            # discarded — no fill on the last bar.
            if self._pending_order is not None and current_bar is not None:
                fill: FillEvent = self._broker.execute(
                    order=self._pending_order,
                    next_bar=current_bar,
                )
                self._portfolio.on_fill(fill)
                self._pending_order = None

            # ---- Step 2: update portfolio ATR buffer and equity curve ----
            # Always called regardless of warm-up state so that ATR is
            # fully converged by the time the first trade fires.
            self._portfolio.on_bar(prev_bar)

            # ---- Step 3: feed bar to strategy ----------------------------
            signal: SignalEvent | None = self._strategy.on_bar(prev_bar)

            # ---- Step 4: act on signal only after warm-up is complete ----
            in_warm_up: bool = self.bar_count < self._warm_up_bars

            if signal is not None and not in_warm_up:
                order: OrderEvent | None = self._portfolio.on_signal(
                    signal=signal,
                    current_bar=prev_bar,
                )
                if order is not None:
                    # Store the order — it will be filled at the NEXT bar's open.
                    self._pending_order = order

            self.bar_count += 1

            # ---- Step 5: advance or terminate ----------------------------
            if current_bar is None:
                # End of feed reached. Any pending order is discarded since
                # there is no next bar to fill on.
                self._pending_order = None
                break

            prev_bar = current_bar

