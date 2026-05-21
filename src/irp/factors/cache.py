"""Parquet-based cache for cross-section snapshots and backtest results.

No DB access.

Layout:
    data/factor_cache/
        A/  YYYY-MM-DD.parquet          # cross-section snapshots
        Q/  YYYY-MM-DD.parquet
        backtest/
            <factor>_h<horizon>_<variant>_<freq>_<start>_<end>/
                ic.parquet              # ic_series as single-column DataFrame
                qcr.parquet             # quintile_cumret DataFrame
                meta.parquet            # mean_ic, ic_tstat, n_dates (one row)
"""
import datetime
import logging
import shutil
from pathlib import Path

import pandas as pd

from irp.core.config import config

logger = logging.getLogger(__name__)

_CACHE_ROOT: Path = Path(config.data.root_dir) / 'factor_cache'


def _path(as_of_date: datetime.date, variant: str) -> Path:
    return _CACHE_ROOT / variant / f'{as_of_date.isoformat()}.parquet'


def load(as_of_date: datetime.date, variant: str) -> pd.DataFrame | None:
    p = _path(as_of_date, variant)
    return pd.read_parquet(p) if p.exists() else None


def store(as_of_date: datetime.date, variant: str, df: pd.DataFrame) -> None:
    p = _path(as_of_date, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)


def clear(variant: str | None = None) -> int:
    """Delete cached snapshots. Returns number of files removed."""
    root = _CACHE_ROOT / variant if variant else _CACHE_ROOT
    if not root.exists():
        return 0
    n = sum(1 for _ in root.rglob('*.parquet'))
    shutil.rmtree(root)
    return n


def _bt_path(
    factor: str,
    horizon_days: int,
    variant: str,
    freq: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> Path:
    name = f'{factor}_h{horizon_days}_{variant}_{freq}_{start_date.isoformat()}_{end_date.isoformat()}'
    return _CACHE_ROOT / 'backtest' / name


def load_backtest(
    factor: str,
    horizon_days: int,
    variant: str,
    freq: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> dict | None:
    d = _bt_path(factor, horizon_days, variant, freq, start_date, end_date)
    if not d.exists():
        return None
    ic = pd.read_parquet(d / 'ic.parquet')['ic']
    ic.index = pd.to_datetime(ic.index).date
    qcr = pd.read_parquet(d / 'qcr.parquet')
    qcr.index = pd.to_datetime(qcr.index).date
    meta = pd.read_parquet(d / 'meta.parquet').iloc[0]
    return {
        'ic_series': ic,
        'quintile_cumret': qcr,
        'mean_ic': float(meta['mean_ic']),
        'ic_tstat': float(meta['ic_tstat']),
        'n_dates': int(meta['n_dates']),
    }


def store_backtest(
    factor: str,
    horizon_days: int,
    variant: str,
    freq: str,
    start_date: datetime.date,
    end_date: datetime.date,
    result: dict,
) -> None:
    d = _bt_path(factor, horizon_days, variant, freq, start_date, end_date)
    d.mkdir(parents=True, exist_ok=True)
    result['ic_series'].rename('ic').to_frame().to_parquet(d / 'ic.parquet')
    result['quintile_cumret'].to_parquet(d / 'qcr.parquet')
    pd.DataFrame([{
        'mean_ic': result['mean_ic'],
        'ic_tstat': result['ic_tstat'],
        'n_dates': result['n_dates'],
    }]).to_parquet(d / 'meta.parquet')


def precompute_all(
    start_date: datetime.date,
    end_date: datetime.date,
    variants: list[str] | None = None,
    freq: str = 'QE',
    force: bool = False,
) -> int:
    """Compute and cache all cross-section snapshots for a date range.

    Fetches raw data once per variant; loops over rebalance dates in memory.
    Skips dates already cached unless force=True.
    Returns number of new snapshots written.
    """
    import pandas as pd
    from irp.factors.compute import _cross_section_from_raw
    from irp.query.simfin import fundamentals
    from irp.query.yahoo import prices as yahoo_prices

    if variants is None:
        variants = ['A', 'Q']

    rebalance_dates = [ts.date() for ts in pd.date_range(start_date, end_date, freq=freq)]
    n_total = len(rebalance_dates)
    written = 0
    for variant in variants:
        cached_count = sum(1 for d in rebalance_dates if _path(d, variant).exists())
        todo = [d for d in rebalance_dates if force or not _path(d, variant).exists()]
        logger.info(
            f'variant {variant}: {cached_count}/{n_total} already cached, '
            f'{len(todo)} to compute'
        )
        if not todo:
            continue
        logger.info(f'variant {variant}: fetching raw data...')
        raw_income   = fundamentals(None, 'income',   variant)
        raw_balance  = fundamentals(None, 'balance',  variant)
        raw_cashflow = fundamentals(None, 'cashflow', variant)
        raw_prices   = yahoo_prices(None)
        logger.info(f'variant {variant}: raw data ready, computing snapshots...')
        for i, d in enumerate(todo, 1):
            xs = _cross_section_from_raw(
                raw_income, raw_balance, raw_cashflow, raw_prices, d, variant
            )
            if not xs.empty:
                store(d, variant, xs)
                written += 1
            logger.info(f'variant {variant}: {i}/{len(todo)}  {d}  ({len(xs)} tickers)')
        logger.info(f'variant {variant}: done — {written} new snapshots written')
    return written
