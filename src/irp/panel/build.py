"""Materialize DuckDB raw tables into parquet panels.

Run once after each ETL update. Output:

    <data_root>/panel/
    ├── prices.parquet            long format, sorted (Ticker, Date)
    ├── income_A.parquet          long, sorted (Ticker, eff_date), with computed eff_date
    ├── income_Q.parquet
    ├── balance_A.parquet
    ├── balance_Q.parquet
    ├── cashflow_A.parquet
    └── cashflow_Q.parquet

eff_date = COALESCE(Publish Date, Report Date + 60 days) — the PIT cutoff.
Fundamentals tables keep only the columns the factor engine uses (skinnier than raw).
"""

import logging
from pathlib import Path
from typing import Literal

import polars as pl

from irp.core.config import config
from irp.core.db import db

logger = logging.getLogger(__name__)


# Columns kept in fundamentals panels (skinnier than raw — only what factors need)
_INCOME_COLS = [
    'Ticker',
    'Report Date',
    'Publish Date',
    'Period',
    'Revenue',
    'Gross Profit',
    'Operating Income (Loss)',
    'Net Income',
    'Interest Expense, Net',
    'Depreciation & Amortization',
    'Shares (Diluted)',
]
_BALANCE_COLS = [
    'Ticker',
    'Report Date',
    'Publish Date',
    'Period',
    'Total Assets',
    'Total Equity',
    'Total Liabilities',
    'Total Current Assets',
    'Total Current Liabilities',
    'Cash, Cash Equivalents & Short Term Investments',
    'Short Term Debt',
    'Long Term Debt',
    'Shares (Diluted)',
]
_CASHFLOW_COLS = [
    'Ticker',
    'Report Date',
    'Publish Date',
    'Period',
    'Net Cash from Operating Activities',
    'Net Cash from Investing Activities',
    'Depreciation & Amortization',
]

_FUND_COLS = {
    'income': _INCOME_COLS,
    'balance': _BALANCE_COLS,
    'cashflow': _CASHFLOW_COLS,
}

# Restated panels keep only the columns needed for revision factors
_INCOME_RESTATED_COLS = ['Ticker', 'Report Date', 'Restated Date', 'Period', 'Revenue', 'Net Income']
_BALANCE_RESTATED_COLS = ['Ticker', 'Report Date', 'Restated Date', 'Period', 'Total Assets', 'Total Equity']
_CASHFLOW_RESTATED_COLS = ['Ticker', 'Report Date', 'Restated Date', 'Period', 'Net Cash from Operating Activities']

_FUND_RESTATED_COLS = {
    'income': _INCOME_RESTATED_COLS,
    'balance': _BALANCE_RESTATED_COLS,
    'cashflow': _CASHFLOW_RESTATED_COLS,
}


def _panel_dir() -> Path:
    d = config.data.root_dir / 'panel'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _quote(c: str) -> str:
    return f'"{c}"'


def build_prices_panel() -> Path:
    """Write `prices.parquet` — long format, sorted (Ticker, Date).

    Schema: Ticker (str), Date (date), Close (f64), Volume (i64).
    Skips Open/High/Low (not used by factor engine).
    """
    conn = db()
    out = _panel_dir() / 'prices.parquet'
    logger.info(f'materializing prices panel → {out}')
    arrow = conn.execute("""
        SELECT Ticker, Date, Close, Volume
        FROM yahoo_prices
        ORDER BY Ticker, Date
    """).arrow()
    df: pl.DataFrame = pl.from_arrow(arrow)  # type: ignore[assignment]
    df.write_parquet(out, compression='zstd', statistics=True)
    logger.info(f'prices panel: {len(df):,} rows, {df["Ticker"].n_unique():,} tickers')
    return out


def build_fundamentals_panel(
    stmt: Literal['income', 'balance', 'cashflow'],
    variant: Literal['A', 'Q'],
) -> Path:
    """Write `<stmt>_<variant>.parquet` — long format with eff_date column.

    eff_date = COALESCE(Publish Date, Report Date + 60 days).
    Sorted by (Ticker, eff_date) for fast PIT slicing.
    """
    cols = _FUND_COLS[stmt]
    cols_sql = ', '.join(_quote(c) for c in cols)
    conn = db()
    arrow = conn.execute(f"""
        SELECT {cols_sql},
               COALESCE("Publish Date", "Report Date" + INTERVAL '60' DAY) AS eff_date
        FROM {stmt}
        WHERE Period = '{variant}'
        ORDER BY Ticker, eff_date
    """).arrow()
    df: pl.DataFrame = pl.from_arrow(arrow)  # type: ignore[assignment]
    out = _panel_dir() / f'{stmt}_{variant}.parquet'
    df.write_parquet(out, compression='zstd', statistics=True)
    logger.info(
        f'{stmt}_{variant} panel: {len(df):,} rows, {df["Ticker"].n_unique():,} tickers → {out}'
    )
    return out


def build_fundamentals_panel_restated(
    stmt: Literal['income', 'balance', 'cashflow'],
    variant: Literal['A', 'Q'],
) -> Path:
    """Write `<stmt>_restated_<variant>.parquet` using Restated Date as eff_date.

    eff_date = COALESCE("Restated Date", "Publish Date") — the date the corrected
    value became publicly available (PIT cutoff for revision signals).
    """
    cols = _FUND_RESTATED_COLS[stmt]
    cols_sql = ', '.join(_quote(c) for c in cols)
    conn = db()
    arrow = conn.execute(f"""
        SELECT {cols_sql},
               COALESCE("Restated Date", "Publish Date") AS eff_date
        FROM {stmt}_restated
        WHERE Period = '{variant}'
        ORDER BY Ticker, eff_date
    """).arrow()
    df: pl.DataFrame = pl.from_arrow(arrow)  # type: ignore[assignment]
    out = _panel_dir() / f'{stmt}_restated_{variant}.parquet'
    df.write_parquet(out, compression='zstd', statistics=True)
    logger.info(
        f'{stmt}_restated_{variant} panel: {len(df):,} rows, {df["Ticker"].n_unique():,} tickers → {out}'
    )
    return out


def build_panels() -> list[Path]:
    """Materialize all panels. Idempotent — overwrites existing files."""
    outs = [build_prices_panel()]
    for stmt in ('income', 'balance', 'cashflow'):
        for variant in ('A', 'Q'):
            outs.append(build_fundamentals_panel(stmt, variant))
            outs.append(build_fundamentals_panel_restated(stmt, variant))
    logger.info(f'panel build complete: {len(outs)} files')
    return outs


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s'
    )
    build_panels()
