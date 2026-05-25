"""Parquet-based cache for cross-section snapshots and backtest results.

No DB access.

Layout:
    data/factor_cache/
        A/  YYYY-MM-DD.parquet          # cross-section snapshots
        Q/  YYYY-MM-DD.parquet
        backtest/
            <factor>_h<horizon>_<variant>_<freq>_<start>_<end>/
                ic.parquet              # ic_series as single-column DataFrame
                qcr.parquet             # quintile_cumret DataFrame (gross)
                ew.parquet              # ew_cumret Series
                ls.parquet              # ls_cumret Series (Q1-Q5 long-short)
                meta.parquet            # mean_ic, ic_tstat, n_dates (one row)
                meta2.parquet           # extended metrics: icir, turnover, quintile risk
"""
import datetime
import logging
import shutil
from pathlib import Path
from typing import Literal

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


_QUINTILE_LABELS = [f'Q{i + 1}' for i in range(5)]
_NAN = float('nan')


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
    result: dict = {
        'ic_series': ic,
        'quintile_cumret': qcr,
        'quintile_cumret_net': None,
        'ew_cumret': pd.Series(dtype=float),
        'ls_cumret': pd.Series(dtype=float),
        'mean_ic': float(meta['mean_ic']),
        'ic_tstat': float(meta['ic_tstat']),
        'icir': _NAN,
        'n_dates': int(meta['n_dates']),
        'quintile_ann_ret': {},
        'quintile_ann_vol': {},
        'quintile_sharpe': {},
        'quintile_max_dd': {},
        'turnover_q1': pd.Series(dtype=float),
        'turnover_q5': pd.Series(dtype=float),
        'mean_turnover_q1': _NAN,
        'mean_turnover_q5': _NAN,
    }
    # Extended metrics (written by new store_backtest; absent in old cache entries)
    if (d / 'ew.parquet').exists():
        ew = pd.read_parquet(d / 'ew.parquet')['ew']
        ew.index = pd.to_datetime(ew.index).date
        result['ew_cumret'] = ew
    if (d / 'ls.parquet').exists():
        ls = pd.read_parquet(d / 'ls.parquet')['ls']
        ls.index = pd.to_datetime(ls.index).date
        result['ls_cumret'] = ls
    if (d / 'meta2.parquet').exists():
        m2 = pd.read_parquet(d / 'meta2.parquet').iloc[0]
        result['icir'] = float(m2.get('icir', _NAN))
        result['mean_turnover_q1'] = float(m2.get('mean_turnover_q1', _NAN))
        result['mean_turnover_q5'] = float(m2.get('mean_turnover_q5', _NAN))
        for pfx, key in [
            ('ann_ret', 'quintile_ann_ret'),
            ('ann_vol', 'quintile_ann_vol'),
            ('sharpe',  'quintile_sharpe'),
            ('max_dd',  'quintile_max_dd'),
        ]:
            result[key] = {q: float(m2.get(f'{pfx}_{q}', _NAN)) for q in _QUINTILE_LABELS}
    return result


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

    ew = result.get('ew_cumret', pd.Series(dtype=float))
    if not ew.empty:
        ew.rename('ew').to_frame().to_parquet(d / 'ew.parquet')
    ls = result.get('ls_cumret', pd.Series(dtype=float))
    if not ls.empty:
        ls.rename('ls').to_frame().to_parquet(d / 'ls.parquet')

    meta2: dict = {
        'icir': result.get('icir', _NAN),
        'mean_turnover_q1': result.get('mean_turnover_q1', _NAN),
        'mean_turnover_q5': result.get('mean_turnover_q5', _NAN),
    }
    for pfx, key in [
        ('ann_ret', 'quintile_ann_ret'),
        ('ann_vol', 'quintile_ann_vol'),
        ('sharpe',  'quintile_sharpe'),
        ('max_dd',  'quintile_max_dd'),
    ]:
        qd = result.get(key, {})
        for q in _QUINTILE_LABELS:
            meta2[f'{pfx}_{q}'] = qd.get(q, _NAN)
    pd.DataFrame([meta2]).to_parquet(d / 'meta2.parquet')


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
    import pandas as pd
    from irp.factors.compute import _compute_and_cache

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
        logger.info(f'variant {variant}: computing {len(todo)} snapshots via SQL...')
        computed = _compute_and_cache(todo, variant, tickers=None)
        written += len(computed)
        logger.info(f'variant {variant}: done — {written} new snapshots written')
    return written
