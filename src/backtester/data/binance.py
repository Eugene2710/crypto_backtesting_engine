"""
data/binance.py
---------------
Binance REST API adapter implementing the DataFeed interface.

Uses `crypto_indicators.binance_api.get_klines` (async, retried) to fetch
OHLCV bars and converts each raw dict into a MarketEvent ready for the engine.

Usage
-----
    feed = BinanceDataFeed(symbol="BTCUSDT", interval="1d", limit=500)
    await feed.load()
    while (bar := feed.next_bar()) is not None:
        ...
"""

from tenacity import retry, wait_fixed, stop_after_attempt

from crypto_indicators.binance_api import get_klines

from src.backtester.data.base import DataFeed
from src.backtester.data.events import MarketEvent


class BinanceDataFeed(DataFeed):
    """
    Fetches OHLCV bars from the Binance public REST API (no API key required).

    Parameters
    ----------
    symbol:   Binance trading pair, e.g. "BTCUSDT"
    interval: Kline interval string, e.g. "1d", "4h", "1h"
    limit:    Number of bars to fetch (max 1000 per Binance API).
              Use ≥500 for daily bars to get enough history for walk-forward.
    """

    def __init__(self, symbol: str, interval: str, limit: int = 500) -> None:
        self._symbol:   str = symbol
        self._interval: str = interval
        self._limit:    int = limit

        # Internal bar store — populated by load()
        self._bars:   list[MarketEvent] = []
        self._cursor: int = 0  # points to the next bar to be yielded

    # ------------------------------------------------------------------
    # DataFeed interface — lifecycle
    # ------------------------------------------------------------------

    @retry(
        wait=wait_fixed(0.01),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def load(self) -> None:
        """
        Fetch bars from Binance and populate the internal bar list.

        Bars are returned by get_klines in ascending order (oldest first),
        which is what we need for strict chronological replay.
        """
        raw: list[dict[str, float]] = await get_klines(
            symbol=self._symbol,
            interval=self._interval,
            limit=self._limit,
        )

        # Convert each raw dict to a typed, validated MarketEvent.
        # bar_index is the position within this loaded history window —
        # used by the engine to enforce the no-lookahead guarantee.
        self._bars = [
            MarketEvent(
                symbol=self._symbol,
                timestamp=int(row["open_time"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                bar_index=i,
            )
            for i, row in enumerate(raw)
        ]
        self._cursor = 0

    # ------------------------------------------------------------------
    # DataFeed interface — iteration
    # ------------------------------------------------------------------

    def next_bar(self) -> MarketEvent | None:
        """Yield the next bar in chronological order, or None when done."""
        if self._cursor >= len(self._bars):
            return None
        bar: MarketEvent = self._bars[self._cursor]
        self._cursor += 1
        return bar

    def __len__(self) -> int:
        return len(self._bars)

    # ------------------------------------------------------------------
    # DataFeed interface — inspection
    # ------------------------------------------------------------------

    def get_bar(self, index: int) -> MarketEvent:
        """Return bar at 0-based index. Raises IndexError if out of range."""
        return self._bars[index]

    def reset(self) -> None:
        """Reset the cursor to bar 0 without re-fetching from the network."""
        self._cursor = 0

    # ------------------------------------------------------------------
    # Slicing helper — used by walk-forward module
    # ------------------------------------------------------------------

    def slice(self, start: int, end: int) -> list[MarketEvent]:
        """
        Return bars[start:end] without advancing the main cursor.

        The walk-forward module uses this to carve out in-sample and
        out-of-sample windows from the same loaded history.

        Parameters
        ----------
        start: inclusive 0-based index
        end:   exclusive 0-based index
        """
        return self._bars[start:end]
