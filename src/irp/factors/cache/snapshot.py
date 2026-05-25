"""Parquet snapshot cache for cross-section factor DataFrames.

One file per (variant, as_of_date) under `data/factor_cache/<variant>/`.
Writes are full overwrites; misses recomputed from the panel layer.
"""
import datetime
import logging
import shutil
from pathlib import Path
from typing import Literal

import pandas as pd

from irp.core.config import config

logger = logging.getLogger(__name__)

CACHE_ROOT: Path = Path(config.data.root_dir) / 'factor_cache'


def _path(as_of_date: datetime.date, variant: str) -> Path:
    return CACHE_ROOT / variant / f'{as_of_date.isoformat()}.parquet'


def load(as_of_date: datetime.date, variant: str) -> pd.DataFrame | None:
    p = _path(as_of_date, variant)
    return pd.read_parquet(p) if p.exists() else None


def store(as_of_date: datetime.date, variant: str, df: pd.DataFrame) -> None:
    p = _path(as_of_date, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)


def clear(variant: str | None = None) -> int:
    """Delete cached snapshots. Returns number of files removed."""
    root = CACHE_ROOT / variant if variant else CACHE_ROOT
    if not root.exists():
        return 0
    n = sum(1 for _ in root.rglob('*.parquet'))
    shutil.rmtree(root)
    return n


def precompute_all(
    start_date: datetime.date,
    end_date: datetime.date,
    variants: list[Literal['A', 'Q']] | None = None,
    freq: str = 'QE',
    force: bool = False,
) -> int:
    """Compute and cache all cross-section snapshots for a date range.

    Fetches raw data once per variant; loops over rebalance dates in memory.
    Skips dates already cached unless force=True.
    Returns number of new snapshots written.
    """
    from irp.factors.data_loaders import compute_and_cache

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
        logger.info(f'variant {variant}: computing {len(todo)} snapshots...')
        computed = compute_and_cache(todo, variant, tickers=None)
        written += len(computed)
        logger.info(f'variant {variant}: done — {written} new snapshots written')
    return written
