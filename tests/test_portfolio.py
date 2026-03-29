"""
test_portfolio.py
-----------------
Phase 5 unit tests — Portfolio P&L, cash tracking, and trade records.

These tests verify that after a buy-then-sell round trip:
  1. Net P&L equals gross gain minus round-trip fees
  2. Cash balance is correctly updated at each step
  3. Position quantity is zero after exit
  4. A completed Trade record is created with correct fields
  5. Equity is marked-to-market correctly while a position is open

Core P&L identity being tested
-------------------------------
  Gross P&L  = (exit_price - entry_price) × quantity
  Entry fee  = entry_price × quantity × fee_rate
  Exit fee   = exit_price  × quantity × fee_rate
  Net P&L    = gross_pnl - entry_fee - exit_fee

Example (used throughout these tests):
  Buy  1 BTC at $100, fee = 100 × 1 × 0.001 = $0.10
  Sell 1 BTC at $110, fee = 110 × 1 × 0.001 = $0.11
  Gross P&L  = (110 - 100) × 1 = $10.00
  Net P&L    = $10.00 - $0.10 - $0.11 = $9.79

These are pure unit tests: no strategy, no engine, no network calls.
Only Portfolio, SimulatedBroker, and synthetic FillEvents are involved.
"""
from tests.helpers import make_bar
from src.backtester.broker import SimulatedBroker
from src.backtester.data.events import OrderEvent, OrderSide
from src.backtester.portfolio import Portfolio, Trade


# ---------------------------------------------------------------------------
# Shared fill construction helper
# ---------------------------------------------------------------------------

def execute_round_trip(
    portfolio:   Portfolio,
    broker:      SimulatedBroker,
    buy_price:   float,
    sell_price:  float,
    quantity:    float,
) -> None:
    """
    Execute a BUY then SELL via the broker and apply both fills
    to the portfolio. Used as the Arrange + Act shared step.
    """
    buy_bar   = make_bar(1, open_=buy_price,  high=buy_price,  low=buy_price,  close=buy_price)
    sell_bar  = make_bar(2, open_=sell_price, high=sell_price, low=sell_price, close=sell_price)

    buy_order  = OrderEvent(symbol="BTCUSDT", timestamp=0, side=OrderSide.BUY,  quantity=quantity)
    sell_order = OrderEvent(symbol="BTCUSDT", timestamp=1, side=OrderSide.SELL, quantity=quantity)

    buy_fill  = broker.execute(buy_order,  buy_bar)
    sell_fill = broker.execute(sell_order, sell_bar)

    portfolio.on_fill(buy_fill)
    portfolio.on_fill(sell_fill)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPortfolioPnL:

    def test_net_pnl_equals_gross_gain_minus_round_trip_fees(
        self,
        portfolio: Portfolio,
        broker:    SimulatedBroker,
    ) -> None:
        """
        Net P&L must equal gross gain minus the sum of entry and exit fees.

        Hand-calculated:
          gross_pnl  = (110 - 100) × 1 = $10.00
          entry_fee  = 100 × 1 × 0.001 = $0.10
          exit_fee   = 110 × 1 × 0.001 = $0.11
          net_pnl    = 10.00 - 0.10 - 0.11 = $9.79
        """
        # Arrange + Act
        execute_round_trip(portfolio, broker, buy_price=100.0, sell_price=110.0, quantity=1.0)

        # Assert
        trade: Trade = portfolio.trades[0]
        expected_net_pnl: float = 10.0 - (100.0 * 0.001) - (110.0 * 0.001)  # 9.79
        assert abs(trade.net_pnl - expected_net_pnl) < 1e-9, (
            f"Expected net P&L {expected_net_pnl:.6f}, got {trade.net_pnl:.6f}"
        )

    def test_cash_balance_after_buy_decreases_by_cost_plus_fee(
        self,
        portfolio: Portfolio,
        broker:    SimulatedBroker,
    ) -> None:
        """
        After a BUY fill, cash must decrease by (fill_price × quantity + commission).

        Hand-calculated:
          initial cash = 10,000
          buy cost     = 100 × 1 = $100.00
          buy fee      = 100 × 1 × 0.001 = $0.10
          cash after   = 10,000 - 100.10 = $9,899.90
        """
        # Arrange
        buy_bar   = make_bar(1, open_=100.0, high=105.0, low=95.0, close=100.0)
        buy_order = OrderEvent(symbol="BTCUSDT", timestamp=0, side=OrderSide.BUY, quantity=1.0)
        buy_fill  = broker.execute(buy_order, buy_bar)

        # Act
        portfolio.on_fill(buy_fill)

        # Assert
        expected_cash: float = 10_000.0 - (100.0 + 100.0 * 0.001)  # 9,899.90
        assert abs(portfolio.cash - expected_cash) < 1e-9, (
            f"Expected cash {expected_cash:.6f}, got {portfolio.cash:.6f}"
        )

    def test_cash_balance_after_sell_increases_by_proceeds_minus_fee(
        self,
        portfolio: Portfolio,
        broker:    SimulatedBroker,
    ) -> None:
        """
        After a full round trip, cash must equal:
          initial - (buy_cost + buy_fee) + (sell_proceeds - sell_fee)

        Hand-calculated:
          initial cash     = 10,000
          buy cost + fee   = 100.10
          sell proceeds    = 110 × 1 = $110.00
          sell fee         = 110 × 1 × 0.001 = $0.11
          final cash       = 10,000 - 100.10 + 109.89 = $10,009.79
        """
        # Arrange + Act
        execute_round_trip(portfolio, broker, buy_price=100.0, sell_price=110.0, quantity=1.0)

        # Assert
        expected_cash: float = (
            10_000.0
            - (100.0 + 100.0 * 0.001)     # buy cost + buy fee
            + (110.0 - 110.0 * 0.001)     # sell proceeds - sell fee
        )
        assert abs(portfolio.cash - expected_cash) < 1e-9, (
            f"Expected cash {expected_cash:.6f}, got {portfolio.cash:.6f}"
        )

    def test_position_quantity_is_zero_after_full_exit(
        self,
        portfolio: Portfolio,
        broker:    SimulatedBroker,
    ) -> None:
        """
        After a full SELL, position_qty must be exactly 0.0.
        A non-zero value means the portfolio thinks it still holds an asset.
        """
        # Arrange + Act
        execute_round_trip(portfolio, broker, buy_price=100.0, sell_price=110.0, quantity=1.0)

        # Assert
        assert portfolio.position_qty == 0.0, (
            f"Expected position_qty 0.0 after full exit, got {portfolio.position_qty}"
        )

    def test_exactly_one_trade_record_created_per_round_trip(
        self,
        portfolio: Portfolio,
        broker:    SimulatedBroker,
    ) -> None:
        """
        A single BUY + SELL round trip must produce exactly one Trade record.
        Two records would mean double-counting; zero means the trade was lost.
        """
        # Arrange + Act
        execute_round_trip(portfolio, broker, buy_price=100.0, sell_price=110.0, quantity=1.0)

        # Assert
        assert len(portfolio.trades) == 1, (
            f"Expected 1 trade record, got {len(portfolio.trades)}"
        )

    def test_trade_record_entry_and_exit_prices_match_fill_prices(
        self,
        portfolio: Portfolio,
        broker:    SimulatedBroker,
    ) -> None:
        """
        The Trade record must store the actual fill prices, not the signal
        bar prices. This ensures P&L calculations use realistic values.
        """
        # Arrange + Act
        execute_round_trip(portfolio, broker, buy_price=100.0, sell_price=110.0, quantity=1.0)

        # Assert
        trade: Trade = portfolio.trades[0]
        assert trade.entry_price == 100.0
        assert trade.exit_price  == 110.0

    def test_equity_mark_to_market_while_position_is_open(
        self,
        portfolio: Portfolio,
        broker:    SimulatedBroker,
    ) -> None:
        """
        While a position is open, equity must reflect the current market
        value of the position, not just cash.

        Hand-calculated after buying 1 BTC at $100:
          cash         = 10,000 - 100.10 = $9,899.90
          position MtM = 1 × $120 (current close) = $120.00
          equity       = $9,899.90 + $120.00 = $10,019.90
        """
        # Arrange — buy 1 BTC at $100
        buy_bar   = make_bar(1, open_=100.0, high=105.0, low=95.0, close=100.0)
        buy_order = OrderEvent(symbol="BTCUSDT", timestamp=0, side=OrderSide.BUY, quantity=1.0)
        buy_fill  = broker.execute(buy_order, buy_bar)
        portfolio.on_fill(buy_fill)

        # Act — price moves to $120, update equity via on_bar
        mark_bar = make_bar(2, open_=120.0, high=125.0, low=115.0, close=120.0)
        portfolio.on_bar(mark_bar)

        # Assert
        expected_cash:   float = 10_000.0 - (100.0 + 100.0 * 0.001)  # 9,899.90
        expected_equity: float = expected_cash + (1.0 * 120.0)        # 10,019.90
        assert abs(portfolio.equity - expected_equity) < 1e-9, (
            f"Expected equity {expected_equity:.6f}, got {portfolio.equity:.6f}"
        )
