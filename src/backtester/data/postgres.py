"""
data/postgres.py
----------------
Postgres / TimescaleDB adapter stub — not yet implemented.

This file exists so the engine can import and reference PostgresDataFeed
without breaking. Swap in the real implementation once the ETL pipeline
is running and the schema is confirmed (TimescaleDB vs plain Postgres TBD).

Relationship to the ETL pipeline
---------------------------------
This adapter is a pure reader. It has no knowledge of how data arrives in
Postgres — that is entirely the ETL pipeline's concern. The only coupling
point is the schema contract: the column names queried here must match
whatever the ETL pipeline writes. Once the schema is finalised, replace
the NotImplementedError bodies with asyncpg / psycopg3 SELECT queries and
the rest of the engine is untouched.

Expected schema (placeholder — finalise when ETL is ready):
    CREATE TABLE ohlcv (
        symbol      TEXT         NOT NULL,
        open_time   TIMESTAMPTZ  NOT NULL,
        open        NUMERIC      NOT NULL,
        high        NUMERIC      NOT NULL,
        low         NUMERIC      NOT NULL,
        close       NUMERIC      NOT NULL,
        volume      NUMERIC      NOT NULL,
        PRIMARY KEY (symbol, open_time)
    );
"""

from __future__ import annotations

from src.backtester.data.base import DataFeed
from src.backtester.events import MarketEvent


class PostgresDataFeed(DataFeed):
    """
    Stub implementation — raises NotImplementedError on all methods.

    Replace with a real asyncpg / psycopg3 implementation once:
      1. The ETL pipeline is populating the OHLCV table.
      2. The database schema (TimescaleDB vs plain Postgres) is confirmed.
    """

    def __init__(self, symbol: str, interval: str, dsn: str) -> None:
        # dsn: standard PostgreSQL connection string
        # e.g. "postgresql://user:pass@localhost:5432/dbname"
        self._symbol:   str = symbol
        self._interval: str = interval
        self._dsn:      str = dsn
        self._bars:     list[MarketEvent] = []
        self._cursor:   int = 0

    async def load(self) -> None:
        raise NotImplementedError(
            "PostgresDataFeed.load() is not implemented yet. "
            "Use BinanceDataFeed until the ETL pipeline is ready."
        )

    def next_bar(self) -> MarketEvent | None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._bars)

    def get_bar(self, index: int) -> MarketEvent:
        raise NotImplementedError

    def reset(self) -> None:
        self._cursor = 0
