"""
test_walk_forward.py
--------------------
Phase 7 integration test — walk-forward validation.

Walk-forward testing is the primary defence against overfitting. A strategy
that only works on the data it was developed on is not robust. Walk-forward
tests whether the strategy generalises to unseen data by evaluating
out-of-sample (OOS) performance on windows the strategy never trained on.

Window configuration
--------------------
  Total bars       : 500 BTCUSDT daily bars (from Binance)
  In-sample (IS)   : 200 bars — used to warm up indicators
  Out-of-sample    : 100 bars — performance evaluated here
  Step             : 100 bars — non-overlapping OOS windows
  Warm-up within window: 50 bars (SMA50 minimum)

With 500 bars, step=100, IS=200 this produces 3 OOS windows:
  Window 1: IS bars 0–199,   OOS bars 200–299
  Window 2: IS bars 100–299, OOS bars 300–399
  Window 3: IS bars 200–399, OOS bars 400–499

Pass condition
--------------
OOS Sharpe > 0 on the majority of windows (at least 2 of 3).
A Sharpe > 0 means the strategy made money on unseen data — a necessary
(not sufficient) condition for robustness. A strategy failing this test
is curve-fitted to in-sample data only.

This is a true integration test: it hits the Binance API and exercises
the full engine stack (DataFeed → Engine → Strategy → Portfolio → Broker).
"""
from src.backtester.broker import SimulatedBroker
from src.backtester.data.binance import BinanceDataFeed
from src.backtester.engine import BacktestEngine
from src.backtester.events import MarketEvent
from src.backtester.performance import performance_report
from src.backtester.portfolio import Portfolio
from src.backtester.strategies.ma_crossover import SMACrossoverStrategy


# ---------------------------------------------------------------------------
# Static feed helper — wraps a bar slice as a DataFeed
# ---------------------------------------------------------------------------

class SlicedFeed(BinanceDataFeed):
    """
    A BinanceDataFeed backed by a pre-loaded slice of bars rather than a
    network request. Used to run the engine on a specific walk-forward window
    without re-fetching from the API.
    """

    def __init__(self, bars: list[MarketEvent]) -> None:
        super().__init__(symbol="BTCUSDT", interval="1d", limit=len(bars))
        self._bars   = bars
        self._cursor = 0

    async def load(self) -> None:
        pass  # Already populated from the parent feed slice


# ---------------------------------------------------------------------------
# Walk-forward runner
# ---------------------------------------------------------------------------

def run_oos_window(
    is_bars:  list[MarketEvent],
    oos_bars: list[MarketEvent],
) -> dict[str, float]:
    """
    Run a walk-forward window and return performance metrics for the OOS
    period only.

    Correct walk-forward architecture
    ----------------------------------
    The engine runs on IS + OOS bars combined. warm_up_bars is set to
    len(is_bars) so that:
      - IS bars warm up the strategy indicators fully — no trades placed.
      - OOS bars are evaluated with fully converged indicators — trades fire.

    Only the OOS portion of the equity curve (last len(oos_bars) entries)
    is passed to performance_report so Sharpe and other metrics reflect
    purely out-of-sample performance.

    Running OOS bars alone with a small warm_up would starve SMA50 of
    history, resulting in no trades and a flat equity curve (Sharpe = 0.0).

    Parameters
    ----------
    is_bars  : in-sample bars — used to warm up indicators, no trades counted
    oos_bars : out-of-sample bars — trades and performance measured here
    """
    all_bars: list[MarketEvent] = is_bars + oos_bars

    strategy  = SMACrossoverStrategy(fast_period=20, slow_period=50)
    portfolio = Portfolio(initial_capital=10_000.0, risk_per_trade=0.01)
    broker    = SimulatedBroker(fee_rate=0.001, slippage=0.0)
    feed      = SlicedFeed(all_bars)

    engine = BacktestEngine(
        feed=feed,
        strategy=strategy,
        portfolio=portfolio,
        broker=broker,
        warm_up_bars=len(is_bars),  # IS bars warm up indicators, OOS bars trade
    )
    engine.run()

    # Extract only the OOS portion of the equity curve for evaluation.
    oos_equity_curve = portfolio.equity_curve[-len(oos_bars):]

    # Extract only trades that opened during the OOS window.
    oos_start_ts: int = oos_bars[0].timestamp
    oos_trades = [t for t in portfolio.trades if t.entry_time >= oos_start_ts]

    return performance_report(
        equity_curve=oos_equity_curve,
        trades=oos_trades,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWalkForward:

    def test_oos_sharpe_positive_on_majority_of_windows(
        self, btcusdt_daily_feed: BinanceDataFeed
    ) -> None:
        """
        The SMA 20/50 strategy must produce positive OOS Sharpe on at least
        2 out of 3 walk-forward windows.

        A positive Sharpe on OOS data means the strategy made money on bars
        it never saw during development — the minimum bar for robustness.
        Failing this test means the strategy is likely curve-fitted.
        """
        # Arrange
        btcusdt_daily_feed.reset()
        all_bars: list[MarketEvent] = [
            btcusdt_daily_feed.get_bar(i) for i in range(len(btcusdt_daily_feed))
        ]

        # in_sample_size also serves as warm_up_bars inside run_oos_window —
        # the 200 IS bars warm up the strategy indicators before OOS trading begins.
        in_sample_size: int = 200
        oos_size:       int = 100
        step:           int = 100

        oos_results:   list[float] = []
        start: int = 0

        while start + in_sample_size + oos_size <= len(all_bars):
            oos_start: int = start + in_sample_size
            oos_end:   int = oos_start + oos_size

            is_bars:  list[MarketEvent] = all_bars[start:oos_start]
            oos_bars: list[MarketEvent] = all_bars[oos_start:oos_end]

            # Act — warm up on IS bars, evaluate on OOS bars
            report = run_oos_window(is_bars=is_bars, oos_bars=oos_bars)
            oos_results.append(report["sharpe_ratio"])

            start += step

        # Assert — need at least 2 windows to draw any conclusions
        assert len(oos_results) >= 2, (
            f"Only {len(oos_results)} OOS windows generated — "
            "increase data limit or reduce window sizes."
        )

        positive_count: int = sum(1 for sr in oos_results if sr > 0)
        majority: int = (len(oos_results) // 2) + 1

        assert positive_count >= majority, (
            f"OOS Sharpe > 0 in only {positive_count}/{len(oos_results)} windows "
            f"(need majority = {majority}). "
            f"Sharpe values: {[round(sr, 3) for sr in oos_results]}"
        )

    def test_walk_forward_produces_expected_number_of_windows(
        self, btcusdt_daily_feed: BinanceDataFeed
    ) -> None:
        """
        With 500 bars, IS=200, OOS=100, step=100, exactly 3 windows must be
        produced. A different count indicates the windowing logic is wrong.
        """
        # Arrange
        btcusdt_daily_feed.reset()
        total_bars:      int = len(btcusdt_daily_feed)  # 500
        in_sample_size:  int = 200
        oos_size:        int = 100
        step:            int = 100

        # Act — count windows without running the full backtest
        window_count: int = 0
        start: int = 0
        while start + in_sample_size + oos_size <= total_bars:
            window_count += 1
            start += step

        # Assert
        assert window_count == 3, (
            f"Expected 3 walk-forward windows, got {window_count}."
        )
