"""
events.py
---------
All event types flowing through the engine's event queue.

Event flow per bar:
  MarketEvent → Strategy emits SignalEvent
  SignalEvent  → Portfolio emits OrderEvent
  OrderEvent   → Broker emits FillEvent
  FillEvent    → Portfolio updates positions / equity

Using Pydantic BaseModel gives us:
  - Runtime type validation and coercion (e.g. raw API ints/floats are
    validated on construction rather than silently accepted as wrong types)
  - .model_dump() / .model_dump_json() for free serialisation
  - Clear schema documentation for future MCP tool integration
"""

from __future__ import annotations
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Direction / side enums
# ---------------------------------------------------------------------------

class SignalDirection(str, Enum):
    """Direction a strategy wants to trade."""
    LONG = "LONG"
    EXIT = "EXIT"


class OrderSide(str, Enum):
    """Side of a broker order."""
    BUY  = "BUY"
    SELL = "SELL"


# ---------------------------------------------------------------------------
# Market event — one OHLCV bar has been consumed
# ---------------------------------------------------------------------------

class MarketEvent(BaseModel):
    """
    Emitted by the DataFeed each time a new bar is made available to the engine.
    The strategy receives this and may decide to emit a SignalEvent.

    bar_index is the 0-based position within the full history slice visible to
    the engine at this point — used to enforce the no-lookahead guarantee.
    """
    event_type: Literal["MARKET"] = "MARKET"

    symbol:    str
    timestamp: int    # open_time in milliseconds (UTC)
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    bar_index: int = 0


# ---------------------------------------------------------------------------
# Signal event — strategy declares directional intent
# ---------------------------------------------------------------------------

class SignalEvent(BaseModel):
    """
    Emitted by a Strategy when a trading signal fires.
    The Portfolio converts this into an OrderEvent with a concrete quantity.

    strength is an optional [0, 1] multiplier; 1.0 means full ATR-sized position.
    """
    event_type: Literal["SIGNAL"] = "SIGNAL"

    symbol:    str
    timestamp: int               # open_time of the bar that triggered the signal
    direction: SignalDirection
    strength:  float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Order event — portfolio requests a trade execution
# ---------------------------------------------------------------------------

class OrderEvent(BaseModel):
    """
    Emitted by the Portfolio after computing ATR-based position size.
    The SimulatedBroker converts this into a FillEvent on the next bar's open.
    """
    event_type: Literal["ORDER"] = "ORDER"

    symbol:    str
    timestamp: int        # open_time of the bar the order was raised on
    side:      OrderSide
    quantity:  float      # base-asset units (e.g. BTC); always > 0


# ---------------------------------------------------------------------------
# Fill event — broker confirms trade execution
# ---------------------------------------------------------------------------

class FillEvent(BaseModel):
    """
    Emitted by the SimulatedBroker once an order is filled.

    fill_price is the next bar's open (conservative, realistic assumption).
    commission and slippage are absolute dollar amounts so the Portfolio can
    compute exact net P&L without knowing the fee schedule itself.
    """
    event_type: Literal["FILL"] = "FILL"

    symbol:     str
    timestamp:  int       # open_time of the fill bar (next bar after signal bar)
    side:       OrderSide
    quantity:   float
    fill_price: float     # next bar open
    commission: float     # dollar cost of taker fee (default 0.1% per side)
    slippage:   float     # dollar slippage (default 0)
