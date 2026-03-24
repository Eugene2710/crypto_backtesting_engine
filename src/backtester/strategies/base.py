"""
strategies/base.py
------------------
Abstract Strategy interface.

Every concrete strategy receives MarketEvents one bar at a time and may
emit zero or one SignalEvent per bar. The strategy has access to all bars
up to and including the current bar — never beyond it.
"""

from abc import ABC, abstractmethod

from src.backtester.events import MarketEvent, SignalEvent


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    The engine calls `on_bar(event)` for every bar in the history window.
    The strategy maintains its own internal state (e.g. rolling price
    history, indicator values) and returns a SignalEvent when a trade
    signal fires, or None when no action is warranted.

    No-lookahead contract
    ---------------------
    `on_bar` receives exactly the bars up to index N. The strategy must
    never store a reference to the DataFeed or access bar N+1 directly.
    Compliance is enforced structurally: the engine only passes the
    current MarketEvent, not the full feed.
    """

    @abstractmethod
    def on_bar(self, event: MarketEvent) -> SignalEvent | None:
        """
        Process one bar and optionally emit a signal.

        Parameters
        ----------
        event: the current bar's MarketEvent (bar index N)

        Returns
        -------
        SignalEvent if a signal fires this bar, else None.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Reset all internal state (price history, indicator buffers, etc.)
        back to the same condition as a freshly constructed instance.

        Called by the walk-forward module before each new window so that
        indicator warm-up state does not bleed across windows.
        """
        ...
