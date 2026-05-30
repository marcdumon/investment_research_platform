"""Panel loaders with in-process caching.

Price panel is stored long-format on disk and built as a dense numpy
(dates × tickers) matrix on first access. We avoid `polars.pivot` because
it took ~20 min on a 37M × 12K matrix; direct numpy fancy-indexing does
the same work in seconds. Fundamentals are small and stay in polars.
"""
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from irp.core.config import config

logger = logging.getLogger(__name__)


def panel_root() -> Path:
    """Where panel parquets live."""
    return config.data.root_dir / 'panel'


@dataclass(frozen=True)
class PricePanel:
    """Dense wide price panel: dates × tickers numpy matrix + index arrays.

    Attributes
    ----------
    dates    : numpy datetime64[D] array (n_dates,)         — sorted ascending
    tickers  : numpy str array        (n_tickers,)          — sorted ascending
    values   : numpy float32 matrix   (n_dates, n_tickers)  — NaN where missing
    ticker_to_idx : dict[str, int]    — O(1) ticker lookup
    """
    dates: np.ndarray
    tickers: np.ndarray
    values: np.ndarray
    ticker_to_idx: dict


@lru_cache(maxsize=4)
def load_prices_wide(metric: Literal['Close', 'Volume'] = 'Close') -> PricePanel:
    """Dense (Date × Ticker) price matrix. Cached in-process.

    Read long-format parquet, build a dense float32 matrix via numpy
    fancy-indexing. NaN where no observation. ~1.2GB for 25K × 12K Close panel.
    """
    src = panel_root() / 'prices.parquet'
    t0 = time.time()
    logger.debug(f'loading prices panel: {src}')
    long = pl.read_parquet(src, columns=['Ticker', 'Date', metric])

    dates_unique  = long['Date'].unique().sort()
    tickers_unique = long['Ticker'].unique().sort()
    dates_np   = dates_unique.to_numpy()
    tickers_np = tickers_unique.to_numpy()

    # Per-row index lookup via join with row index frames
    date_idx_lf = dates_unique.to_frame().with_row_index('_date_idx').rename({'Date': 'Date'})
    tick_idx_lf = tickers_unique.to_frame().with_row_index('_tick_idx').rename({'Ticker': 'Ticker'})
    joined = long.join(date_idx_lf, on='Date').join(tick_idx_lf, on='Ticker')
    di = joined['_date_idx'].to_numpy()
    ti = joined['_tick_idx'].to_numpy()
    vals = joined[metric].to_numpy().astype('float32')

    n_d, n_t = len(dates_np), len(tickers_np)
    matrix = np.full((n_d, n_t), np.nan, dtype='float32')
    matrix[di, ti] = vals

    panel = PricePanel(
        dates=dates_np,
        tickers=tickers_np,
        values=matrix,
        ticker_to_idx={t: i for i, t in enumerate(tickers_np)},
    )
    logger.info(
        f'prices wide [{metric}]: {n_d:,} dates × {n_t:,} tickers, '
        f'{matrix.nbytes / 1e6:.0f}MB, built in {time.time() - t0:.1f}s'
    )
    return panel


@lru_cache(maxsize=12)
def load_fundamentals(
    stmt: Literal['income', 'balance', 'cashflow'],
    variant: Literal['A', 'Q'],
) -> pl.DataFrame:
    """Long-format fundamentals panel with eff_date column, sorted (Ticker, eff_date)."""
    src = panel_root() / f'{stmt}_{variant}.parquet'
    df = pl.read_parquet(src)
    logger.debug(f'loaded {stmt}_{variant}: {len(df):,} rows')
    return df


@lru_cache(maxsize=12)
def _load_fundamentals_restated(
    stmt: Literal['income', 'balance', 'cashflow'],
    variant: Literal['A', 'Q'],
) -> pl.DataFrame:
    """Restated fundamentals panel — eff_date = Restated Date (PIT for revision signals)."""
    src = panel_root() / f'{stmt}_restated_{variant}.parquet'
    df = pl.read_parquet(src)
    logger.debug(f'loaded {stmt}_restated_{variant}: {len(df):,} rows')
    return df


@lru_cache(maxsize=1)
def _load_ta_panel() -> pl.DataFrame:
    """Long-format TA panel: (Ticker, Date, indicator...). Sorted by Date for PIT lookup.

    Returns an empty DataFrame if ta_panel.parquet does not exist (build_ta_panel() not yet run).
    """
    src = panel_root() / 'ta_panel.parquet'
    if not src.exists():
        logger.warning('ta_panel.parquet not found — run build_panels() to precompute TA')
        return pl.DataFrame()
    df = pl.read_parquet(src).sort('Date')  # sort once; group_by.last() relies on this order
    logger.debug(f'loaded ta_panel: {len(df):,} rows, {df["Ticker"].n_unique():,} tickers')
    return df


def clear_cache() -> None:
    """Drop in-memory panel caches (call after rebuild)."""
    load_prices_wide.cache_clear()
    load_fundamentals.cache_clear()
    _load_fundamentals_restated.cache_clear()
    _load_ta_panel.cache_clear()
