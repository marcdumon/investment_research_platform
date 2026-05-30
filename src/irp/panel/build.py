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

import numpy as np
import pandas as pd
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


def _build_prices_panel() -> Path:
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


def _build_fundamentals_panel(
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


def _build_fundamentals_panel_restated(
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


def _build_ta_panel() -> Path:
    """Write `ta_panel.parquet` — long format, sorted (Ticker, Date).

    Schema: Ticker (str), Date (date), <indicator> (f32) per TaSpec in _TA_SPECS.
    Rows where all indicators are NaN (lookback period) are dropped.
    Re-run this whenever _TA_SPECS changes (add/remove a factor).
    """
    from irp.factors.technical import _TA_SPECS  # lazy — avoids circular import at module level

    ta_names = [s.name for s in _TA_SPECS]
    if not ta_names:
        logger.info('no TA specs registered — skipping ta_panel build')
        return _panel_dir() / 'ta_panel.parquet'

    src = _panel_dir() / 'prices.parquet'
    logger.info(f'building ta_panel ({len(ta_names)} indicators) from {src}')

    prices = pl.read_parquet(src, columns=['Ticker', 'Date', 'Close']).sort(['Ticker', 'Date'])
    groups = prices.partition_by('Ticker', maintain_order=True)

    chunks: list[pl.DataFrame] = []
    n_total = len(groups)
    for i, group in enumerate(groups):
        ticker: str = group['Ticker'][0]
        dates_np: np.ndarray = group['Date'].to_numpy()      # datetime64[D]
        closes = group['Close'].cast(pl.Float64).to_numpy()  # float64
        n = len(closes)
        if n < 2:
            continue

        close_series = pd.Series(closes)
        ta_arrays: dict[str, np.ndarray] = {}
        for s in _TA_SPECS:
            try:
                ta_arrays[s.name] = s.fn(close_series).to_numpy(dtype='float32')
            except Exception:
                ta_arrays[s.name] = np.full(n, np.nan, dtype='float32')

        any_finite = np.zeros(n, dtype=bool)
        for arr in ta_arrays.values():
            any_finite |= np.isfinite(arr)
        if not any_finite.any():
            continue

        chunk = pl.DataFrame({
            'Date': dates_np[any_finite],
            **{name: arr[any_finite] for name, arr in ta_arrays.items()},
        }).with_columns(pl.lit(ticker).alias('Ticker'))
        chunks.append(chunk)

        if (i + 1) % 2000 == 0:
            logger.info(f'ta_panel: {i + 1}/{n_total} tickers processed')

    if chunks:
        ta_df = pl.concat(chunks).sort(['Ticker', 'Date'])
    else:
        schema = {'Ticker': pl.Utf8, 'Date': pl.Date, **{n: pl.Float32 for n in ta_names}}
        ta_df = pl.DataFrame(schema=schema)

    out = _panel_dir() / 'ta_panel.parquet'
    ta_df.write_parquet(out, compression='zstd', statistics=True)
    logger.info(f'ta_panel: {len(ta_df):,} rows, {ta_df["Ticker"].n_unique():,} tickers → {out}')
    return out


def build_panels() -> list[Path]:
    """Materialize all panels. Idempotent — overwrites existing files."""
    outs = [_build_prices_panel()]
    for stmt in ('income', 'balance', 'cashflow'):
        for variant in ('A', 'Q'):
            outs.append(_build_fundamentals_panel(stmt, variant))
            outs.append(_build_fundamentals_panel_restated(stmt, variant))
    outs.append(_build_ta_panel())
    logger.info(f'panel build complete: {len(outs)} files')
    return outs


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s'
    )
    build_panels()
