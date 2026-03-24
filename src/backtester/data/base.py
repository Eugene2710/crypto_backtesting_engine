"""
data/base.py
------------
Abstract DataFeed interface.

All concrete adapters (Binance, Postgres, CSV, …) must implement this
interface so the engine never depends on a specific data source.
"""

from abc import ABC, abstractmethod
from tenacity import retry, wait_fixed, stop_after_attempt

from src.backtester.events import MarketEvent


class DataFeed(ABC):
    """
    Abstract base class for all data adapters.

    The engine calls `load()` once during initialisation to fetch the full
    history, then iterates over bars via `__iter__` / `next_bar()`.

    Design constraints
    ------------------
    - `load()` must return bars in strictly ascending chronological order.
    - The concrete implementation is responsible for any warm-up bars needed
      by indicators (e.g. 200 bars before the strategy start date).
    - No bar at index N may contain information from bar N+1 or later.
      This is enforced structurally: the feed yields one bar at a time and
      the engine never allows the strategy to peek ahead.
    """

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    @abstractmethod
    @retry(
        wait=wait_fixed(0.01),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def load(self) -> None:
        """
        Fetch / read all bars from the underlying source and store them
        internally so that `next_bar()` can iterate over them.

        Must be called (and awaited) before any calls to `next_bar()`.
        """
        ...

    # -----------------------------------------------------------------
    # Iteration
    # -----------------------------------------------------------------

    @abstractmethod
    def next_bar(self) -> MarketEvent | None:
        """
        Return the next MarketEvent in chronological order, or None when
        the history is exhausted.

        Each call advances an internal cursor by exactly one bar.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Total number of bars available after `load()` has been called."""
        ...

    # -----------------------------------------------------------------
    # Inspection helpers (used by performance and walk-forward modules)
    # -----------------------------------------------------------------

    @abstractmethod
    def get_bar(self, index: int) -> MarketEvent:
        """
        Return the bar at the given 0-based index.

        Raises IndexError if index is out of range.
        This method exists for testing / walk-forward slicing only.
        The engine itself never calls it during the main event loop.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the internal cursor to bar 0.

        Allows the same loaded feed to be replayed (e.g. for walk-forward
        windows) without re-fetching data from the network.
        """
        ...
