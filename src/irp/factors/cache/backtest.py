"""Parquet cache for backtest results (IC series, quintile cumret, metrics).

Layout: `data/factor_cache/backtest/<key>/` with multiple parquets per result.
Key encodes factor, horizon, variant, freq, date range — composite backtests
and cost-adjusted runs bypass the cache.

`schema_version` lives in `meta.parquet`. Reads are tolerant to old entries
written without `schema_version` (treated as v0; extended fields fill with NaN).
Writes always stamp the current version.
"""
import datetime
import logging
from pathlib import Path

import pandas as pd

from irp.factors.cache import snapshot as _snapshot
from irp.factors.models import BacktestResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_QUINTILE_LABELS = [f'Q{i + 1}' for i in range(5)]
_NAN = float('nan')


def _path(
    factor: str,
    horizon_days: int,
    variant: str,
    freq: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> Path:
    name = f'{factor}_h{horizon_days}_{variant}_{freq}_{start_date.isoformat()}_{end_date.isoformat()}'
    return _snapshot.CACHE_ROOT / 'backtest' / name


def load_backtest(
    factor: str,
    horizon_days: int,
    variant: str,
    freq: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> BacktestResult | None:
    d = _path(factor, horizon_days, variant, freq, start_date, end_date)
    if not d.exists():
        return None
    ic = pd.read_parquet(d / 'ic.parquet')['ic']
    ic.index = pd.to_datetime(ic.index).date
    qcr = pd.read_parquet(d / 'qcr.parquet')
    qcr.index = pd.to_datetime(qcr.index).date
    meta = pd.read_parquet(d / 'meta.parquet').iloc[0]
    result = BacktestResult(
        ic_series=ic,
        quintile_cumret=qcr,
        mean_ic=float(meta['mean_ic']),
        ic_tstat=float(meta['ic_tstat']),
        n_dates=int(meta['n_dates']),
    )
    # Extended metrics (written by current store_backtest; absent in pre-v1 entries)
    if (d / 'ew.parquet').exists():
        ew = pd.read_parquet(d / 'ew.parquet')['ew']
        ew.index = pd.to_datetime(ew.index).date
        result.ew_cumret = ew
    if (d / 'ls.parquet').exists():
        ls = pd.read_parquet(d / 'ls.parquet')['ls']
        ls.index = pd.to_datetime(ls.index).date
        result.ls_cumret = ls
    if (d / 'meta2.parquet').exists():
        m2 = pd.read_parquet(d / 'meta2.parquet').iloc[0]
        result.icir = float(m2.get('icir', _NAN))
        result.mean_turnover_q1 = float(m2.get('mean_turnover_q1', _NAN))
        result.mean_turnover_q5 = float(m2.get('mean_turnover_q5', _NAN))
        for pfx, key in [
            ('ann_ret', 'quintile_ann_ret'),
            ('ann_vol', 'quintile_ann_vol'),
            ('sharpe',  'quintile_sharpe'),
            ('max_dd',  'quintile_max_dd'),
        ]:
            setattr(result, key, {q: float(m2.get(f'{pfx}_{q}', _NAN)) for q in _QUINTILE_LABELS})
    return result


def store_backtest(
    factor: str,
    horizon_days: int,
    variant: str,
    freq: str,
    start_date: datetime.date,
    end_date: datetime.date,
    result: BacktestResult,
) -> None:
    d = _path(factor, horizon_days, variant, freq, start_date, end_date)
    d.mkdir(parents=True, exist_ok=True)
    result.ic_series.rename('ic').to_frame().to_parquet(d / 'ic.parquet')
    result.quintile_cumret.to_parquet(d / 'qcr.parquet')
    pd.DataFrame([{
        'schema_version': SCHEMA_VERSION,
        'mean_ic': result.mean_ic,
        'ic_tstat': result.ic_tstat,
        'n_dates': result.n_dates,
    }]).to_parquet(d / 'meta.parquet')

    if not result.ew_cumret.empty:
        result.ew_cumret.rename('ew').to_frame().to_parquet(d / 'ew.parquet')
    if not result.ls_cumret.empty:
        result.ls_cumret.rename('ls').to_frame().to_parquet(d / 'ls.parquet')

    meta2: dict = {
        'icir': result.icir,
        'mean_turnover_q1': result.mean_turnover_q1,
        'mean_turnover_q5': result.mean_turnover_q5,
    }
    for pfx, key in [
        ('ann_ret', 'quintile_ann_ret'),
        ('ann_vol', 'quintile_ann_vol'),
        ('sharpe',  'quintile_sharpe'),
        ('max_dd',  'quintile_max_dd'),
    ]:
        qd = getattr(result, key) or {}
        for q in _QUINTILE_LABELS:
            meta2[f'{pfx}_{q}'] = qd.get(q, _NAN)
    pd.DataFrame([meta2]).to_parquet(d / 'meta2.parquet')
