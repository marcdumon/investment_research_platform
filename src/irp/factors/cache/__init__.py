"""Factor-result cache layer.

Layout:
    data/factor_cache/
        A/  YYYY-MM-DD.parquet              # cross-section snapshots (snapshot.py)
        Q/  YYYY-MM-DD.parquet
        backtest/
            <factor>_h<horizon>_<variant>_<freq>_<start>_<end>/
                ic.parquet                  # ic_series
                qcr.parquet                 # quintile cum returns
                ew.parquet                  # ew_cumret
                ls.parquet                  # ls_cumret (Q1 - Q5)
                meta.parquet                # mean_ic, ic_tstat, n_dates, schema_version
                meta2.parquet               # icir, turnover, quintile risk

Two concerns, two modules: `snapshot` for cross-section parquets,
`backtest` for assembled result parquets. Re-exported from here for
backwards compatibility — historic imports `from irp.factors import cache`
or `from irp.factors.cache import load` keep working.
"""
from irp.factors.cache.backtest import (
    SCHEMA_VERSION,
    load_backtest,
    store_backtest,
)
from irp.factors.cache.snapshot import (
    CACHE_ROOT,
    clear,
    load,
    precompute_all,
    store,
)

__all__ = [
    'CACHE_ROOT',
    'SCHEMA_VERSION',
    'clear',
    'load',
    'load_backtest',
    'precompute_all',
    'store',
    'store_backtest',
]
