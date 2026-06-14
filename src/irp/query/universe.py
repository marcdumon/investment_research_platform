"""Provider-agnostic ticker universe — read accessor.

The `universe` table has one row per instrument:

  Ticker        VARCHAR  — canonical key (Stooq SrcId stripped of exchange suffix)
  Market        VARCHAR  — market category from Stooq zip structure
  stooq_ticker  VARCHAR  — full Stooq source ticker (e.g. AAPL.US)
  yahoo_ticker  VARCHAR  — yfinance symbol (e.g. EURUSD=X); NULL if no Yahoo equivalent

Seeding + refreshing the table (with the market-priority dedup) now lives in
`dataload.universe`; the irp CLI / UI call it via `irp.core.ingest_context`.
This module is the read-only accessor, consistent with irp.query.stooq /
irp.query.yahoo / irp.query.catalog.
"""
import pandas as pd

from irp.query._common import db, ticker_filter


def universe(tickers: str | list[str] | None = None) -> pd.DataFrame:
    """Provider-agnostic ticker universe with per-provider symbols."""
    clause, params = ticker_filter(tickers)
    where = f'WHERE {clause}' if clause else ''
    return db().execute(f'SELECT * FROM main.universe {where} ORDER BY Ticker', params).df()
