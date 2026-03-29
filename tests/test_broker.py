"""
test_broker.py
--------------
Phase 4 unit tests — SimulatedBroker fill correctness.

These tests verify that the broker:
  1. Fills orders at the next bar's open price (not the signal bar's close)
  2. Calculates commission on the actual fill value
  3. Applies slippage adversely (buys fill higher, sells fill lower)
  4. Returns a FillEvent with all fields correctly populated

These are pure unit tests: only the broker and synthetic events are involved.
No strategy, portfolio, or network calls.

Principles applied
------------------
- Each test verifies one specific aspect of the broker's fill logic.
- Expected values are hand-calculated in comments so failures are
  immediately traceable to a specific arithmetic step.
- Floating point comparisons use a tight tolerance (1e-9) rather than
  equality to handle binary representation rounding.
"""
from tests.helpers import make_bar
from src.backtester.broker import SimulatedBroker
from src.backtester.events import FillEvent, OrderEvent, OrderSide


class TestSimulatedBrokerFill:

    def test_buy_fills_at_next_bar_open_price(
        self, broker: SimulatedBroker
    ) -> None:
        """
        A BUY order must fill at the next bar's open, not the current bar's
        close. Filling at the close would constitute lookahead bias.
        """
        # Arrange
        order    = OrderEvent(symbol="BTCUSDT", timestamp=0, side=OrderSide.BUY, quantity=0.1)
        next_bar = make_bar(1, open_=50_000.0, high=51_000.0, low=49_000.0, close=50_500.0)

        # Act
        fill: FillEvent = broker.execute(order, next_bar)

        # Assert
        assert fill.fill_price == 50_000.0, (
            f"Expected fill at next bar open 50000.0, got {fill.fill_price}"
        )

    def test_sell_fills_at_next_bar_open_price(
        self, broker: SimulatedBroker
    ) -> None:
        """
        A SELL order must also fill at the next bar's open price.
        """
        # Arrange
        order    = OrderEvent(symbol="BTCUSDT", timestamp=0, side=OrderSide.SELL, quantity=0.1)
        next_bar = make_bar(1, open_=55_000.0, high=56_000.0, low=54_000.0, close=55_500.0)

        # Act
        fill: FillEvent = broker.execute(order, next_bar)

        # Assert
        assert fill.fill_price == 55_000.0, (
            f"Expected fill at next bar open 55000.0, got {fill.fill_price}"
        )

    def test_buy_commission_equals_fill_value_times_fee_rate(
        self, broker: SimulatedBroker
    ) -> None:
        """
        Commission = fill_price × quantity × fee_rate.

        Hand-calculated:
          fill_price = 50,000
          quantity   = 0.1
          fee_rate   = 0.001
          commission = 50,000 × 0.1 × 0.001 = $5.00
        """
        # Arrange
        order    = OrderEvent(symbol="BTCUSDT", timestamp=0, side=OrderSide.BUY, quantity=0.1)
        next_bar = make_bar(1, open_=50_000.0, high=51_000.0, low=49_000.0, close=50_500.0)

        # Act
        fill: FillEvent = broker.execute(order, next_bar)

        # Assert
        expected_commission: float = 50_000.0 * 0.1 * 0.001  # = 5.00
        assert abs(fill.commission - expected_commission) < 1e-9, (
            f"Expected commission {expected_commission}, got {fill.commission}"
        )

    def test_sell_commission_equals_fill_value_times_fee_rate(
        self, broker: SimulatedBroker
    ) -> None:
        """
        Commission on SELL = fill_price × quantity × fee_rate.

        Hand-calculated:
          fill_price = 55,000
          quantity   = 0.1
          fee_rate   = 0.001
          commission = 55,000 × 0.1 × 0.001 = $5.50
        """
        # Arrange
        order    = OrderEvent(symbol="BTCUSDT", timestamp=0, side=OrderSide.SELL, quantity=0.1)
        next_bar = make_bar(1, open_=55_000.0, high=56_000.0, low=54_000.0, close=55_500.0)

        # Act
        fill: FillEvent = broker.execute(order, next_bar)

        # Assert
        expected_commission: float = 55_000.0 * 0.1 * 0.001  # = 5.50
        assert abs(fill.commission - expected_commission) < 1e-9, (
            f"Expected commission {expected_commission}, got {fill.commission}"
        )

    def test_buy_slippage_increases_fill_price(self) -> None:
        """
        Slippage is applied adversely: a BUY fills at a higher price.

        Hand-calculated:
          raw_open   = 50,000
          slippage   = 0.001 (0.1%)
          fill_price = 50,000 × (1 + 0.001) = 50,050
        """
        # Arrange
        broker_with_slippage = SimulatedBroker(fee_rate=0.001, slippage=0.001)
        order    = OrderEvent(symbol="BTCUSDT", timestamp=0, side=OrderSide.BUY, quantity=0.1)
        next_bar = make_bar(1, open_=50_000.0, high=51_000.0, low=49_000.0, close=50_500.0)

        # Act
        fill: FillEvent = broker_with_slippage.execute(order, next_bar)

        # Assert
        expected_fill_price: float = 50_000.0 * (1 + 0.001)  # = 50,050
        assert abs(fill.fill_price - expected_fill_price) < 1e-9, (
            f"Expected fill price {expected_fill_price}, got {fill.fill_price}"
        )

    def test_sell_slippage_decreases_fill_price(self) -> None:
        """
        Slippage is applied adversely: a SELL fills at a lower price.

        Hand-calculated:
          raw_open   = 55,000
          slippage   = 0.001 (0.1%)
          fill_price = 55,000 × (1 - 0.001) = 54,945
        """
        # Arrange
        broker_with_slippage = SimulatedBroker(fee_rate=0.001, slippage=0.001)
        order    = OrderEvent(symbol="BTCUSDT", timestamp=0, side=OrderSide.SELL, quantity=0.1)
        next_bar = make_bar(1, open_=55_000.0, high=56_000.0, low=54_000.0, close=55_500.0)

        # Act
        fill: FillEvent = broker_with_slippage.execute(order, next_bar)

        # Assert
        expected_fill_price: float = 55_000.0 * (1 - 0.001)  # = 54,945
        assert abs(fill.fill_price - expected_fill_price) < 1e-9, (
            f"Expected fill price {expected_fill_price}, got {fill.fill_price}"
        )

    def test_fill_event_carries_correct_symbol_side_and_quantity(
        self, broker: SimulatedBroker
    ) -> None:
        """
        The FillEvent must echo back the symbol, side, and quantity from
        the original OrderEvent unchanged.
        """
        # Arrange
        order    = OrderEvent(symbol="BTCUSDT", timestamp=0, side=OrderSide.BUY, quantity=0.25)
        next_bar = make_bar(1, open_=50_000.0, high=51_000.0, low=49_000.0, close=50_500.0)

        # Act
        fill: FillEvent = broker.execute(order, next_bar)

        # Assert
        assert fill.symbol   == "BTCUSDT"
        assert fill.side     == OrderSide.BUY
        assert fill.quantity == 0.25
