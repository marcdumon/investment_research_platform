"""Backtest orchestration: rebalance schedule → cross-sections → IC math.

Three entry points:
- `run_backtest(factor, ...)`           single-factor IC + quintile analysis
- `run_composite_backtest(weights, ...)` weighted multi-factor signal
- `run_factor_decay(factor, horizons, ...)` mean IC + ICIR per horizon

Data loading lives in `data_loaders.py`; this module owns the temporal
loop, composite assembly, and result-level caching.
"""
import datetime
import logging
from typing import Literal

import pandas as pd

from irp.factors import cache as _cache
from irp.factors.backtest import compute_backtest
from irp.factors.data_loaders import _load_cross_sections as load_cross_sections
from irp.factors.models import BacktestResult
from irp.panel import forward_returns_panel

logger = logging.getLogger(__name__)


def _rebalance_dates(
    start_date: datetime.date,
    end_date: datetime.date,
    freq: Literal['Q', 'A'],
) -> list[datetime.date]:
    pd_freq = 'QE' if freq == 'Q' else 'YE'
    return [ts.date() for ts in pd.date_range(start_date, end_date, freq=pd_freq)]


def run_composite_backtest(
    weights: dict[str, float],
    horizon_days: int,
    start_date: datetime.date,
    end_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    freq: Literal['Q', 'A'] = 'Q',
    normalize: str = 'zscore',
    use_sector_neutral: bool = False,
    tickers: list[str] | None = None,
    cost_bps: float = 0,
) -> BacktestResult:
    """IC series and quintile cumulative returns for a multi-factor composite.

    Builds a composite signal at each rebalance date from cached cross-sections,
    then evaluates it with the same IC / quintile methodology as run_backtest().
    Composite results are not cached (weights can vary freely).
    """
    from irp.features.composite import build_composite

    rebalance_dates = _rebalance_dates(start_date, end_date, freq)
    cross_sections_raw = load_cross_sections(rebalance_dates, variant, tickers)

    sector = None
    if use_sector_neutral:
        from irp.query.simfin import sector_map
        sector = sector_map()

    cross_sections_enriched: dict[datetime.date, pd.DataFrame] = {}
    for d, xs in cross_sections_raw.items():
        if tickers is not None:
            xs = xs[xs.index.isin(tickers)]
        score = build_composite(xs, weights, normalize=normalize, sector=sector)
        xs_out = xs.copy()
        xs_out['__composite__'] = score
        cross_sections_enriched[d] = xs_out

    if not cross_sections_enriched:
        return BacktestResult()

    fwd_returns = forward_returns_panel(rebalance_dates, horizon_days, tickers)
    logger.info('composite backtest: computing IC series and quintile returns...')
    result = compute_backtest(
        '__composite__', cross_sections_enriched, fwd_returns,
        horizon_days=horizon_days, cost_bps=cost_bps,
    )
    logger.info(
        f'composite backtest: done — mean IC={result.mean_ic:.3f}, '
        f't-stat={result.ic_tstat:.2f}, ICIR={result.icir:.2f}, '
        f'n={result.n_dates} dates'
    )
    return result


def run_backtest(
    factor: str,
    horizon_days: int,
    start_date: datetime.date,
    end_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    tickers: list[str] | None = None,
    freq: Literal['Q', 'A'] = 'Q',
    cost_bps: float = 0,
) -> BacktestResult:
    """IC series and quintile cumulative returns for one factor over a date range.

    Full-universe + zero-cost runs are served from the backtest result cache
    when available.
    """
    if tickers is None and cost_bps == 0:
        cached = _cache.load_backtest(factor, horizon_days, variant, freq, start_date, end_date)
        if cached is not None:
            logger.info(f'backtest cache hit: {factor} h={horizon_days} {variant}/{freq} {start_date}–{end_date}')
            return cached

    rebalance_dates = _rebalance_dates(start_date, end_date, freq)
    cross_sections = load_cross_sections(rebalance_dates, variant, tickers)

    n_total = len(rebalance_dates)
    n_loaded = len(cross_sections)
    logger.info(
        f'backtest {factor} h={horizon_days} {variant}/{freq} {start_date}–{end_date}: '
        f'{n_loaded}/{n_total} cross-sections ready'
    )

    logger.info(f'backtest {factor}: computing forward returns...')
    fwd_returns = forward_returns_panel(rebalance_dates, horizon_days, tickers)
    logger.info(f'backtest {factor}: computing IC series and quintile returns...')
    result = compute_backtest(
        factor, cross_sections, fwd_returns,
        horizon_days=horizon_days, cost_bps=cost_bps,
    )
    if tickers is None and cost_bps == 0 and result.n_dates > 0:
        _cache.store_backtest(factor, horizon_days, variant, freq, start_date, end_date, result)
        logger.info(
            f'backtest {factor}: done — mean IC={result.mean_ic:.3f}, '
            f't-stat={result.ic_tstat:.2f}, ICIR={result.icir:.2f}, '
            f'n={result.n_dates} dates (result cached)'
        )
    return result


def run_factor_decay(
    factor: str,
    horizons: list[int],
    start_date: datetime.date,
    end_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    freq: Literal['Q', 'A'] = 'Q',
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Mean IC and ICIR across multiple forward-return horizons.

    Reuses cached cross-sections across all horizons; only forward-return
    computation differs per horizon.
    """
    rebalance_dates = _rebalance_dates(start_date, end_date, freq)
    cross_sections = load_cross_sections(rebalance_dates, variant, tickers)

    if tickers is not None:
        cross_sections = {d: xs[xs.index.isin(tickers)] for d, xs in cross_sections.items()}

    rows = []
    for h in sorted(horizons):
        fwd = forward_returns_panel(rebalance_dates, h, tickers)
        res = compute_backtest(factor, cross_sections, fwd, horizon_days=h)
        rows.append({
            'horizon': h,
            'mean_ic': res.mean_ic,
            'icir': res.icir,
            'n_dates': res.n_dates,
        })
        logger.info(f'factor decay {factor} h={h}: mean_ic={res.mean_ic:.3f}, ICIR={res.icir:.2f}')

    return pd.DataFrame(rows)
