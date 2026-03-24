"""
performance.py
--------------
Computes all required performance metrics from a completed backtest.

All functions are pure — they take data in, return numbers out, with no
side effects. This makes them independently testable (Phase 6 integration
test feeds a known equity curve and asserts exact values).

Metrics computed
----------------
  Sharpe ratio          — annualised (primary optimisation metric)
  Sortino ratio         — penalises downside volatility only
  Calmar ratio          — annualised return / max drawdown
  Max drawdown          — peak-to-trough decline as a fraction
  Max drawdown duration — bars from peak to trough of worst drawdown
  Win rate              — fraction of trades that closed profitable
  Profit factor         — gross wins / gross losses
  Average win           — mean net P&L of winning trades
  Average loss          — mean net P&L of losing trades (positive number)
  Strategy return       — total return over the backtest period
  Benchmark return      — buy-and-hold return over the same period

Annualisation
-------------
Crypto trades 24/7, so we use 365 periods per year for daily bars.
Pass `periods_per_year=365*6` for 4H bars or `365*24` for 1H bars.
"""

import math

from src.backtester.portfolio import Trade


# Crypto trades every day — no weekends or holidays.
PERIODS_PER_YEAR_DAILY: int = 365


# ---------------------------------------------------------------------------
# Return series
# ---------------------------------------------------------------------------

def equity_to_returns(equity_curve: list[tuple[int, float]]) -> list[float]:
    """
    Convert an equity curve into period-over-period returns.

        r_t = (equity_t - equity_{t-1}) / equity_{t-1}

    Returns a list one element shorter than the equity curve.
    """
    returns: list[float] = []
    for i in range(1, len(equity_curve)):
        prev_equity: float = equity_curve[i - 1][1]
        curr_equity: float = equity_curve[i][1]
        r: float = (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        returns.append(r)
    return returns


def equity_to_returns_series(
    equity_curve: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """
    Convert an equity curve into a timestamped return series for plotting.

    Each entry is (timestamp_ms, return) where timestamp_ms is the bar's
    Unix millisecond timestamp — the same format as the equity curve.

    The return at position i is paired with equity_curve[i+1]'s timestamp
    because the return is computed from bar i to bar i+1, and the result
    belongs to bar i+1 (the bar where the change was observed).

    Use equity_to_returns() for internal ratio calculations.
    Use this function when you need the time axis for charting.
    """
    returns: list[float] = equity_to_returns(equity_curve)
    # Pair each return with the timestamp of the bar it was observed on (i+1).
    timestamps: list[int] = [equity_curve[i][0] for i in range(1, len(equity_curve))]
    return list(zip(timestamps, returns))


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

def sharpe_ratio(
    equity_curve: list[tuple[int, float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR_DAILY,
) -> float:
    """
    Annualised Sharpe ratio.

        Sharpe = (mean_excess_return / std_return) × sqrt(periods_per_year)

    where mean_excess_return = mean_return - risk_free_rate_per_period.

    Returns 0.0 if fewer than 2 data points or zero volatility.

    Parameters
    ----------
    risk_free_rate : annualised rate as a decimal (e.g. 0.05 = 5%).
                     Converted to per-period rate internally.
    """
    returns: list[float] = equity_to_returns(equity_curve)
    if len(returns) < 2:
        return 0.0

    rf_per_period: float = risk_free_rate / periods_per_year
    mean_r: float = sum(returns) / len(returns)
    excess_mean: float = mean_r - rf_per_period

    variance: float = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std: float = math.sqrt(variance)

    if std == 0:
        return 0.0

    return (excess_mean / std) * math.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------

def sortino_ratio(
    equity_curve: list[tuple[int, float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR_DAILY,
) -> float:
    """
    Annualised Sortino ratio.

    Same as Sharpe but divides by downside deviation only — returns above
    the risk-free rate are not penalised. Better metric for strategies that
    have frequent small gains and rare large losses.

        Sortino = (mean_excess_return / downside_std) × sqrt(periods_per_year)

    Returns 0.0 if no downside returns exist or fewer than 2 data points.
    """
    returns: list[float] = equity_to_returns(equity_curve)
    if len(returns) < 2:
        return 0.0

    rf_per_period: float = risk_free_rate / periods_per_year
    mean_r: float = sum(returns) / len(returns)
    excess_mean: float = mean_r - rf_per_period

    # Only squared deviations below the risk-free rate contribute.
    downside_sq: list[float] = [
        (r - rf_per_period) ** 2
        for r in returns
        if r < rf_per_period
    ]

    if not downside_sq:
        return 0.0

    downside_std: float = math.sqrt(sum(downside_sq) / len(downside_sq))

    if downside_std == 0:
        return 0.0

    return (excess_mean / downside_std) * math.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

def max_drawdown(equity_curve: list[tuple[int, float]]) -> tuple[float, int]:
    """
    Maximum peak-to-trough drawdown and its duration in bars.

    Drawdown at bar t = (peak_equity_up_to_t - equity_t) / peak_equity_up_to_t

    Returns
    -------
    (max_dd, max_dd_duration) where:
      max_dd          : worst drawdown as a positive fraction (e.g. 0.25 = 25%)
      max_dd_duration : bars from the peak bar to the trough bar of the worst
                        drawdown. If equity never recovered, duration runs to
                        the last bar.
    """
    if len(equity_curve) < 2:
        return 0.0, 0

    peak: float = equity_curve[0][1]
    peak_index: int = 0
    max_dd_value: float = 0.0
    max_dd_duration: int = 0

    for i, (_, equity) in enumerate(equity_curve):
        if equity > peak:
            peak = equity
            peak_index = i

        drawdown: float = (peak - equity) / peak if peak > 0 else 0.0

        if drawdown > max_dd_value:
            max_dd_value = drawdown
            max_dd_duration = i - peak_index

    return max_dd_value, max_dd_duration


# ---------------------------------------------------------------------------
# Calmar ratio
# ---------------------------------------------------------------------------

def calmar_ratio(
    equity_curve: list[tuple[int, float]],
    periods_per_year: int = PERIODS_PER_YEAR_DAILY,
) -> float:
    """
    Calmar ratio = annualised return / max drawdown.

    Measures how much return you earn per unit of drawdown risk.
    A Calmar > 1 means the annualised return exceeds the max drawdown.

    Returns 0.0 if max drawdown is zero or the curve is too short.
    """
    if len(equity_curve) < 2:
        return 0.0

    start_equity: float = equity_curve[0][1]
    end_equity:   float = equity_curve[-1][1]
    n_periods:    int   = len(equity_curve) - 1

    if start_equity <= 0 or n_periods == 0:
        return 0.0

    # Compound annualised growth rate.
    cagr: float = (end_equity / start_equity) ** (periods_per_year / n_periods) - 1

    dd, _ = max_drawdown(equity_curve)

    if dd == 0:
        return 0.0

    return cagr / dd


# ---------------------------------------------------------------------------
# Trade-level metrics
# ---------------------------------------------------------------------------

def win_rate(trades: list[Trade]) -> float:
    """Fraction of completed trades with net_pnl > 0."""
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.net_pnl > 0) / len(trades)


def profit_factor(trades: list[Trade]) -> float:
    """
    Gross profit / gross loss.

    A value > 1 means the strategy made more in winners than it lost in losers.
    Returns 0.0 if there are no losing trades.
    """
    gross_profit: float = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss:   float = sum(-t.net_pnl for t in trades if t.net_pnl < 0)

    if gross_loss == 0:
        return 0.0

    return gross_profit / gross_loss


def average_win(trades: list[Trade]) -> float:
    """Mean net P&L of winning trades. Returns 0.0 if no winners."""
    winners: list[float] = [t.net_pnl for t in trades if t.net_pnl > 0]
    return sum(winners) / len(winners) if winners else 0.0


def average_loss(trades: list[Trade]) -> float:
    """
    Mean magnitude of losing trades, returned as a positive number.
    Returns 0.0 if no losers.
    """
    losers: list[float] = [-t.net_pnl for t in trades if t.net_pnl < 0]
    return sum(losers) / len(losers) if losers else 0.0


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def total_return(equity_curve: list[tuple[int, float]]) -> float:
    """
    Total return over the full backtest period as a decimal fraction.

        total_return = (final_equity - initial_equity) / initial_equity
    """
    if len(equity_curve) < 2:
        return 0.0
    start: float = equity_curve[0][1]
    end:   float = equity_curve[-1][1]
    return (end - start) / start if start > 0 else 0.0


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def performance_report(
    equity_curve: list[tuple[int, float]],
    trades: list[Trade],
    benchmark_equity_curve: list[tuple[int, float]] | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR_DAILY,
) -> dict[str, float]:
    """
    Compute all metrics and return them as a flat dictionary.

    Parameters
    ----------
    equity_curve            : strategy equity curve from Portfolio
    trades                  : completed trades from Portfolio
    benchmark_equity_curve  : buy-and-hold equity curve for the same period.
                              Constructed externally by running a fully-invested
                              position through the same bars. If None, benchmark
                              return is omitted (set to 0.0).
    risk_free_rate          : annualised risk-free rate decimal (default 0.0)
    periods_per_year        : 365 for daily, 365*6 for 4H, 365*24 for 1H
    """
    dd, dd_duration = max_drawdown(equity_curve)

    bm_return: float = (
        total_return(benchmark_equity_curve)
        if benchmark_equity_curve is not None
        else 0.0
    )

    return {
        "sharpe_ratio":          sharpe_ratio(equity_curve, risk_free_rate, periods_per_year),
        "sortino_ratio":         sortino_ratio(equity_curve, risk_free_rate, periods_per_year),
        "calmar_ratio":          calmar_ratio(equity_curve, periods_per_year),
        "max_drawdown":          dd,
        "max_drawdown_duration": float(dd_duration),
        "win_rate":              win_rate(trades),
        "profit_factor":         profit_factor(trades),
        "average_win":           average_win(trades),
        "average_loss":          average_loss(trades),
        "total_trades":          float(len(trades)),
        "strategy_return":       total_return(equity_curve),
        "benchmark_return":      bm_return,
    }
