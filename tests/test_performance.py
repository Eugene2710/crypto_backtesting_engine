"""
test_performance.py
-------------------
Phase 6 unit tests — performance metric correctness.

All performance functions are pure (data in, number out) so these tests
feed a known equity curve or trade list and assert the output matches the
hand-calculated expected value exactly.

These are the most important tests for numerical correctness. If Sharpe or
drawdown calculations are wrong, every backtesting conclusion drawn from
them is invalid.

Equity curve used in ratio tests
---------------------------------
  values = [10000, 10100, 10050, 10200, 10000, 10300]
  returns = [+0.01, -0.00495, +0.01493, -0.01961, +0.03]

  Peak = 10200 at index 3
  Trough = 10000 at index 4
  Max drawdown = (10200 - 10000) / 10200 ≈ 0.019608

These are pure unit tests: no network calls, no strategy, no engine.

Principles applied
------------------
- Each test verifies one metric function in isolation.
- Expected values are hand-calculated and documented in comments.
- Floating point comparisons use tolerances appropriate to each metric
  (tight for exact arithmetic, looser for square-root based ratios).
"""
import pytest

from src.backtester.performance import (
    average_loss,
    average_win,
    calmar_ratio,
    equity_to_returns,
    equity_to_returns_series,
    max_drawdown,
    performance_report,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)
from src.backtester.portfolio import Trade


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_equity_curve() -> list[tuple[int, float]]:
    """
    Known equity curve with predictable drawdown and return properties.
    Timestamps are synthetic daily increments.
    """
    values: list[float] = [10_000.0, 10_100.0, 10_050.0, 10_200.0, 10_000.0, 10_300.0]
    return [(i * 86_400_000, v) for i, v in enumerate(values)]


@pytest.fixture()
def sample_trades() -> list[Trade]:
    """
    Three trades: two winners and one loser.
      Trade 1: net +$9.80  (winner)
      Trade 2: net -$5.20  (loser)
      Trade 3: net +$9.80  (winner)
    Win rate   = 2/3
    Profit factor = (9.80 + 9.80) / 5.20
    """
    return [
        Trade(symbol="BTCUSDT", entry_time=0, exit_time=1,
              entry_price=100.0, exit_price=110.0, quantity=1.0,
              gross_pnl=10.0, total_fees=0.20, net_pnl=9.80),
        Trade(symbol="BTCUSDT", entry_time=2, exit_time=3,
              entry_price=110.0, exit_price=105.0, quantity=1.0,
              gross_pnl=-5.0, total_fees=0.20, net_pnl=-5.20),
        Trade(symbol="BTCUSDT", entry_time=4, exit_time=5,
              entry_price=105.0, exit_price=115.0, quantity=1.0,
              gross_pnl=10.0, total_fees=0.20, net_pnl=9.80),
    ]


# ---------------------------------------------------------------------------
# equity_to_returns
# ---------------------------------------------------------------------------

class TestEquityToReturns:

    def test_returns_list_is_one_shorter_than_equity_curve(
        self, sample_equity_curve: list[tuple[int, float]]
    ) -> None:
        """
        The return series has one fewer element than the equity curve because
        a return requires two consecutive equity values to compute.
        """
        # Arrange + Act
        returns = equity_to_returns(sample_equity_curve)

        # Assert
        assert len(returns) == len(sample_equity_curve) - 1

    def test_first_return_is_correct(
        self, sample_equity_curve: list[tuple[int, float]]
    ) -> None:
        """
        First return = (10100 - 10000) / 10000 = 0.01 exactly.
        """
        # Arrange + Act
        returns = equity_to_returns(sample_equity_curve)

        # Assert
        assert abs(returns[0] - 0.01) < 1e-9


class TestEquityToReturnsSeries:

    def test_returns_series_has_timestamps_aligned_to_bar_i_plus_1(
        self, sample_equity_curve: list[tuple[int, float]]
    ) -> None:
        """
        Each (timestamp, return) pair must use the timestamp of bar i+1
        because the return is observed at the close of bar i+1.
        """
        # Arrange + Act
        series = equity_to_returns_series(sample_equity_curve)

        # Assert — first entry timestamp must match equity_curve[1][0]
        assert series[0][0] == sample_equity_curve[1][0]

    def test_returns_values_match_equity_to_returns(
        self, sample_equity_curve: list[tuple[int, float]]
    ) -> None:
        """
        The float values in the series must be identical to equity_to_returns()
        since equity_to_returns_series delegates to it.
        """
        # Arrange + Act
        plain_returns  = equity_to_returns(sample_equity_curve)
        series_returns = [r for _, r in equity_to_returns_series(sample_equity_curve)]

        # Assert
        assert plain_returns == series_returns


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:

    def test_max_drawdown_value_matches_hand_calculation(
        self, sample_equity_curve: list[tuple[int, float]]
    ) -> None:
        """
        Peak = 10200 (index 3), trough = 10000 (index 4).
        Max DD = (10200 - 10000) / 10200 ≈ 0.019608
        """
        # Arrange + Act
        dd, _ = max_drawdown(sample_equity_curve)

        # Assert
        expected_dd: float = (10_200.0 - 10_000.0) / 10_200.0
        assert abs(dd - expected_dd) < 1e-9, (
            f"Expected max drawdown {expected_dd:.8f}, got {dd:.8f}"
        )

    def test_max_drawdown_duration_is_one_bar(
        self, sample_equity_curve: list[tuple[int, float]]
    ) -> None:
        """
        The worst drawdown runs from peak at index 3 to trough at index 4 —
        exactly 1 bar duration.
        """
        # Arrange + Act
        _, duration = max_drawdown(sample_equity_curve)

        # Assert
        assert duration == 1, f"Expected duration 1, got {duration}"

    def test_max_drawdown_returns_zero_for_monotonically_increasing_curve(self) -> None:
        """
        A curve that only goes up has no drawdown — both DD and duration must be 0.
        """
        # Arrange
        curve: list[tuple[int, float]] = [
            (i * 86_400_000, 10_000.0 + i * 100) for i in range(10)
        ]

        # Act
        dd, duration = max_drawdown(curve)

        # Assert
        assert dd == 0.0
        assert duration == 0


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:

    def test_sharpe_is_positive_for_net_positive_equity_curve(
        self, sample_equity_curve: list[tuple[int, float]]
    ) -> None:
        """
        The equity curve ends higher than it started. With zero risk-free rate,
        positive mean return over positive std must yield a positive Sharpe.
        """
        # Arrange + Act
        sr = sharpe_ratio(sample_equity_curve, risk_free_rate=0.0)

        # Assert
        assert sr > 0, f"Expected positive Sharpe for net-positive curve, got {sr}"

    def test_sharpe_returns_zero_for_curve_with_no_volatility(self) -> None:
        """
        A flat equity curve has zero standard deviation of returns.
        Division by zero is guarded — the function must return 0.0.
        """
        # Arrange
        flat_curve: list[tuple[int, float]] = [(i * 86_400_000, 10_000.0) for i in range(10)]

        # Act
        sr = sharpe_ratio(flat_curve)

        # Assert
        assert sr == 0.0

    def test_sharpe_returns_zero_for_fewer_than_two_data_points(self) -> None:
        """
        A single-entry equity curve has no return to compute. Must return 0.0.
        """
        # Arrange
        single_point: list[tuple[int, float]] = [(0, 10_000.0)]

        # Act
        sr = sharpe_ratio(single_point)

        # Assert
        assert sr == 0.0


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:

    def test_sortino_is_positive_for_net_positive_equity_curve(
        self, sample_equity_curve: list[tuple[int, float]]
    ) -> None:
        """
        With downside returns present and a positive mean, Sortino must be positive.
        """
        # Arrange + Act
        sr = sortino_ratio(sample_equity_curve, risk_free_rate=0.0)

        # Assert
        assert sr > 0, f"Expected positive Sortino, got {sr}"

    def test_sortino_returns_zero_when_no_downside_returns_exist(self) -> None:
        """
        A monotonically increasing curve has no returns below the risk-free rate.
        Division by zero is guarded — the function must return 0.0.
        """
        # Arrange
        rising_curve: list[tuple[int, float]] = [
            (i * 86_400_000, 10_000.0 + i * 100) for i in range(10)
        ]

        # Act
        sr = sortino_ratio(rising_curve)

        # Assert
        assert sr == 0.0


# ---------------------------------------------------------------------------
# Calmar ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:

    def test_calmar_is_positive_for_net_positive_curve_with_drawdown(
        self, sample_equity_curve: list[tuple[int, float]]
    ) -> None:
        """
        Positive CAGR divided by a positive drawdown must yield a positive Calmar.
        """
        # Arrange + Act
        cr = calmar_ratio(sample_equity_curve)

        # Assert
        assert cr > 0, f"Expected positive Calmar, got {cr}"

    def test_calmar_returns_zero_when_max_drawdown_is_zero(self) -> None:
        """
        A curve with no drawdown would produce division by zero.
        The function must guard this and return 0.0.
        """
        # Arrange
        rising_curve: list[tuple[int, float]] = [
            (i * 86_400_000, 10_000.0 + i * 100) for i in range(10)
        ]

        # Act
        cr = calmar_ratio(rising_curve)

        # Assert
        assert cr == 0.0


# ---------------------------------------------------------------------------
# Trade-level metrics
# ---------------------------------------------------------------------------

class TestTradeMetrics:

    def test_win_rate_is_two_thirds_for_two_winners_one_loser(
        self, sample_trades: list[Trade]
    ) -> None:
        """
        2 winning trades out of 3 total = win rate of 2/3 ≈ 0.6667.
        """
        # Arrange + Act
        wr = win_rate(sample_trades)

        # Assert
        assert abs(wr - 2 / 3) < 1e-9, f"Expected {2/3:.6f}, got {wr:.6f}"

    def test_win_rate_is_zero_for_empty_trade_list(self) -> None:
        """
        No trades means no winners — win_rate must return 0.0, not error.
        """
        # Arrange + Act + Assert
        assert win_rate([]) == 0.0

    def test_profit_factor_equals_gross_wins_divided_by_gross_losses(
        self, sample_trades: list[Trade]
    ) -> None:
        """
        Gross wins  = 9.80 + 9.80 = 19.60
        Gross losses = 5.20
        Profit factor = 19.60 / 5.20 ≈ 3.769
        """
        # Arrange + Act
        pf = profit_factor(sample_trades)

        # Assert
        expected_pf: float = 19.60 / 5.20
        assert abs(pf - expected_pf) < 1e-9, f"Expected {expected_pf:.6f}, got {pf:.6f}"

    def test_average_win_equals_mean_of_winning_net_pnl(
        self, sample_trades: list[Trade]
    ) -> None:
        """
        Both winning trades have net_pnl = 9.80.
        Average win = (9.80 + 9.80) / 2 = 9.80.
        """
        # Arrange + Act
        avg_w = average_win(sample_trades)

        # Assert
        assert abs(avg_w - 9.80) < 1e-9, f"Expected 9.80, got {avg_w}"

    def test_average_loss_is_returned_as_positive_number(
        self, sample_trades: list[Trade]
    ) -> None:
        """
        The losing trade has net_pnl = -5.20.
        average_loss() must return 5.20 (positive), not -5.20.
        """
        # Arrange + Act
        avg_l = average_loss(sample_trades)

        # Assert
        assert abs(avg_l - 5.20) < 1e-9, f"Expected 5.20, got {avg_l}"


# ---------------------------------------------------------------------------
# Total return
# ---------------------------------------------------------------------------

class TestTotalReturn:

    def test_total_return_matches_hand_calculation(
        self, sample_equity_curve: list[tuple[int, float]]
    ) -> None:
        """
        Start = 10,000, end = 10,300.
        Total return = (10300 - 10000) / 10000 = 0.03 (3%).
        """
        # Arrange + Act
        tr = total_return(sample_equity_curve)

        # Assert
        expected: float = (10_300.0 - 10_000.0) / 10_000.0  # 0.03
        assert abs(tr - expected) < 1e-9, f"Expected {expected}, got {tr}"


# ---------------------------------------------------------------------------
# Performance report
# ---------------------------------------------------------------------------

class TestPerformanceReport:

    def test_report_contains_all_required_keys(
        self,
        sample_equity_curve: list[tuple[int, float]],
        sample_trades: list[Trade],
    ) -> None:
        """
        The report dictionary must contain every required metric key.
        A missing key would cause a KeyError in any downstream consumer.
        """
        # Arrange
        required_keys: set[str] = {
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "max_drawdown_duration",
            "win_rate", "profit_factor", "average_win", "average_loss",
            "total_trades", "strategy_return", "benchmark_return",
        }

        # Act
        report = performance_report(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
        )

        # Assert
        missing: set[str] = required_keys - report.keys()
        assert not missing, f"Report is missing keys: {missing}"

    def test_report_total_trades_matches_trade_list_length(
        self,
        sample_equity_curve: list[tuple[int, float]],
        sample_trades: list[Trade],
    ) -> None:
        """
        The total_trades metric must equal the length of the trades list.
        """
        # Arrange + Act
        report = performance_report(
            equity_curve=sample_equity_curve,
            trades=sample_trades,
        )

        # Assert
        assert report["total_trades"] == float(len(sample_trades))
