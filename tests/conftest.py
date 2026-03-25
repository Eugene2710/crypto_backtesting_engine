"""
conftest.py
-----------
Shared pytest fixtures available to every test file automatically.

Fixture scopes
--------------
- function (default) : fresh instance per test — used for stateful objects
                       like Portfolio and Strategy that mutate during a test.
- module             : one instance per test file — used for the Binance API
                       feed so the network call is made only once per module.

Fixture structure
-----------------
Every fixture follows the try/yield/finally pattern:
  - Setup in try block
  - yield inside try so teardown always runs (pass, fail, or error)
  - finally for unconditional teardown

Principles applied
------------------
- Fixtures own the Arrange step so test bodies focus on Act + Assert only.
- Stateful objects are function-scoped to prevent state leaking between tests.
- The Binance feed is module-scoped so the API is called once per module.
- pytest.fail() inside except gives a clear human-readable failure message
  instead of a raw traceback from deep inside the network stack.
"""

import asyncio
from collections.abc import Generator

import pytest

from src.backtester.broker import SimulatedBroker
from src.backtester.data.binance import BinanceDataFeed
from src.backtester.portfolio import Portfolio
from src.backtester.strategies.ma_crossover import SMACrossoverStrategy
from src.backtester.strategies.macd import MACDCrossoverStrategy


# ---------------------------------------------------------------------------
# Broker fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def broker() -> Generator[SimulatedBroker, None, None]:
    """
    Standard SimulatedBroker: 0.1% taker fee, zero slippage.

    Function-scoped: a fresh instance per test ensures fee_rate and slippage
    cannot be accidentally mutated by one test and affect another.
    """
    instance = SimulatedBroker(fee_rate=0.001, slippage=0.0)
    try:
        yield instance
    finally:
        pass  # SimulatedBroker holds no external resources — nothing to release


# ---------------------------------------------------------------------------
# Portfolio fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def portfolio() -> Generator[Portfolio, None, None]:
    """
    Standard Portfolio: $10,000 capital, 1% risk per trade, ATR(14).

    Function-scoped: Portfolio accumulates cash, trades, and equity curve
    state during a test. Each test must start from a clean slate.
    """
    instance = Portfolio(
        initial_capital=10_000.0,
        risk_per_trade=0.01,
        atr_period=14,
    )
    try:
        yield instance
    finally:
        pass  # In-memory object — no external resources to release


# ---------------------------------------------------------------------------
# Strategy fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sma_strategy() -> Generator[SMACrossoverStrategy, None, None]:
    """
    SMA 20/50 crossover strategy.

    Function-scoped: the strategy accumulates close buffer and crossover
    state bar-by-bar. Tests must not share a partially-warmed strategy.
    """
    instance = SMACrossoverStrategy(fast_period=20, slow_period=50)
    try:
        yield instance
    finally:
        instance.reset()  # Explicitly clear internal state on teardown


@pytest.fixture()
def macd_strategy() -> Generator[MACDCrossoverStrategy, None, None]:
    """
    MACD 12/26/9 crossover strategy.

    Function-scoped for the same reason as sma_strategy — EMA state
    accumulates bar-by-bar and must not bleed between tests.
    """
    instance = MACDCrossoverStrategy(fast_period=12, slow_period=26, signal_period=9)
    try:
        yield instance
    finally:
        instance.reset()  # Explicitly clear EMA and crossover state on teardown


# ---------------------------------------------------------------------------
# Binance feed fixture — module-scoped to avoid repeated API calls
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def btcusdt_daily_feed() -> Generator[BinanceDataFeed, None, None]:
    """
    500 BTCUSDT daily bars fetched from the Binance public REST API.

    Module-scoped: the network request is made exactly once per test module.
    Tests that iterate the feed must call feed.reset() in their Arrange
    section to restore the cursor to bar 0 before iterating.

    pytest.fail() inside except surfaces network failures as a clear,
    human-readable message rather than a raw aiohttp traceback.
    """
    feed = BinanceDataFeed(symbol="BTCUSDT", interval="1d", limit=500)
    try:
        asyncio.run(feed.load())
        yield feed
    except Exception as exc:
        pytest.fail(
            f"Failed to load Binance feed — is the API reachable?\n{exc}"
        )
    finally:
        # Reset the cursor so the feed object is in a clean state after
        # the module finishes, even if a test left it mid-iteration.
        feed.reset()
