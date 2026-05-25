"""Cross-section factor computation orchestrator.

Hot-path factor work runs on `irp.panel` (wide-format parquet panels +
polars/numpy). The single-ticker `ticker_factor_history` path still uses
`irp.factors._pit` and `compute_valuation` etc. for pandas history.
"""
import datetime
import logging
from typing import Literal

import pandas as pd

from irp.core.config import config
from irp.factors import cache as _cache
from irp.factors._cols import REPORT_DATE, TICKER
from irp.factors._pit import pit_latest, pit_prepare, pit_price, pit_ttm
from irp.factors.backtest import compute_backtest
from irp.panel import cross_section_panel, forward_returns_panel
from irp.factors.valuation import compute_valuation
from irp.factors.profitability import compute_profitability
from irp.factors.momentum import compute_momentum
from irp.query.simfin import fundamentals
from irp.query.yahoo import prices as yahoo_prices

logger = logging.getLogger(__name__)


def _compute_and_cache(
    dates: list[datetime.date],
    variant: str,
    tickers: list[str] | None,
    write_cache: bool = True,
) -> dict[datetime.date, pd.DataFrame]:
    """Compute cross-sections for `dates` via SQL; optionally write to cache.

    Returns dict of date → DataFrame (non-empty results only).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(d: datetime.date):
        xs = cross_section_panel(d, variant, tickers)
        return d, xs

    result: dict[datetime.date, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=config.factors.cache_workers) as ex:
        futures = {ex.submit(_one, d): d for d in dates}
        for fut in as_completed(futures):
            d, xs = fut.result()
            if not xs.empty:
                result[d] = xs
                if write_cache:
                    _cache.store(d, variant, xs)
    return result


def cross_section(
    as_of_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Compute all factors cross-sectionally at a point in time.

    Only fundamental data with Report Date <= as_of_date and prices with
    Date <= as_of_date are used, making results PIT-safe.

    Parameters
    ----------
    as_of_date : Snapshot date for the cross-section.
    variant    : 'A' for annual filings, 'Q' for quarterly.
    tickers    : Restrict to specific tickers; None = full universe.

    Returns
    -------
    DataFrame indexed by Ticker with all factor columns.
    Tickers without sufficient data are silently absent.
    """
    if tickers is None:
        cached = _cache.load(as_of_date, variant)
        if cached is not None:
            return cached

    result = cross_section_panel(as_of_date, variant, tickers)
    if tickers is None and not result.empty:
        _cache.store(as_of_date, variant, result)
    return result


def ticker_factor_history(
    ticker: str,
    variant: Literal['A', 'Q'] = 'A',
) -> pd.DataFrame:
    """Factor values at each historical filing date for one ticker.

    Fetches raw data once; computes PIT-aligned factors at each Report Date.
    Returns a DataFrame with columns [Ticker, Report Date, mktcap, pe, pb, ps,
    ev_ebitda, ev_ebit, ev_sales, fcf_yield, gross_margin, op_margin, net_margin,
    roe, roa, roic, fcf_margin], indexed 0..N-1.
    Returns an empty DataFrame if any required data is unavailable.
    """
    raw_income   = fundamentals([ticker], 'income',   variant)
    raw_balance  = fundamentals([ticker], 'balance',  variant)
    raw_cashflow = fundamentals([ticker], 'cashflow', variant)
    raw_prices   = yahoo_prices([ticker])

    if any(df.empty for df in [raw_income, raw_balance, raw_cashflow, raw_prices]):
        return pd.DataFrame()

    report_dates = sorted(raw_income[REPORT_DATE].dropna().unique())
    rows = []
    for rd in report_dates:
        as_of = pd.Timestamp(rd).date()
        inc = pit_ttm(raw_income,   as_of) if variant == 'Q' else pit_latest(raw_income,   as_of)
        bal = pit_latest(raw_balance,  as_of)
        cf  = pit_ttm(raw_cashflow, as_of) if variant == 'Q' else pit_latest(raw_cashflow, as_of)
        px  = pit_price(raw_prices,    as_of)
        if inc.empty or bal.empty or cf.empty or px.empty:
            continue
        row = compute_valuation(inc, bal, cf, px).join(
            compute_profitability(inc, bal, cf), how='outer',
        ).join(
            compute_momentum(raw_prices, as_of), how='outer',
        ).reset_index()
        row[REPORT_DATE] = pd.Timestamp(rd)
        rows.append(row)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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
) -> dict:
    """IC series and quintile cumulative returns for a multi-factor composite.

    Builds a composite signal at each rebalance date from cached cross-sections,
    then evaluates it with the same IC / quintile methodology as run_backtest().

    Parameters
    ----------
    weights           : Factor column → weight mapping (e.g. {'pe': -1, 'roe': 1}).
                        Use PRESETS from irp.features.composite for predefined models.
    horizon_days      : Forward-return horizon in calendar days.
    start_date        : First rebalance date (inclusive).
    end_date          : Last rebalance date (inclusive).
    variant           : 'A' annual or 'Q' quarterly fundamentals.
    freq              : 'Q' quarterly or 'A' annual rebalance schedule.
    normalize         : 'zscore' | 'rank' | 'none'. Applied per factor before weighting.
    use_sector_neutral: Demean within sector before normalizing.

    Returns
    -------
    Same dict as compute_backtest(): ic_series, quintile_cumret, mean_ic, ic_tstat, n_dates.
    Results are NOT cached (composite weights can vary freely).
    """
    from irp.features.composite import build_composite

    pd_freq = 'QE' if freq == 'Q' else 'YE'
    rebalance_dates = [ts.date() for ts in pd.date_range(start_date, end_date, freq=pd_freq)]

    cross_sections_raw: dict[datetime.date, pd.DataFrame] = {}
    dates_to_compute = []
    for d in rebalance_dates:
        cached = _cache.load(d, variant)
        if cached is not None:
            cross_sections_raw[d] = cached
        else:
            dates_to_compute.append(d)

    if dates_to_compute:
        logger.info(f'composite backtest: computing {len(dates_to_compute)} uncached cross-sections')
        cross_sections_raw.update(
            _compute_and_cache(dates_to_compute, variant, tickers)
        )

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
        return {
            'ic_series': pd.Series(dtype=float),
            'quintile_cumret': pd.DataFrame(),
            'mean_ic': float('nan'),
            'ic_tstat': float('nan'),
            'n_dates': 0,
        }

    fwd_returns = forward_returns_panel(rebalance_dates, horizon_days, tickers)
    logger.info('composite backtest: computing IC series and quintile returns...')
    result = compute_backtest(
        '__composite__', cross_sections_enriched, fwd_returns,
        horizon_days=horizon_days, cost_bps=cost_bps,
    )
    logger.info(
        f'composite backtest: done — mean IC={result["mean_ic"]:.3f}, '
        f't-stat={result["ic_tstat"]:.2f}, ICIR={result.get("icir", float("nan")):.2f}, '
        f'n={result["n_dates"]} dates'
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
) -> dict:
    """IC series and quintile cumulative returns for one factor over a date range.

    Fetches raw data once; loops over rebalance dates in memory.

    Parameters
    ----------
    factor       : Factor column name (e.g. 'pe', 'mom_12_1').
    horizon_days : Forward-return horizon in calendar days (e.g. 252 = ~12m).
    start_date   : First rebalance date (inclusive).
    end_date     : Last rebalance date (inclusive).
    variant      : 'A' annual or 'Q' quarterly fundamentals.
    tickers      : Subset of tickers; None = full universe.
    freq         : 'Q' quarterly or 'A' annual rebalance schedule.

    Returns
    -------
    dict from compute_backtest(): ic_series, quintile_cumret, mean_ic, ic_tstat, n_dates.
    """
    if tickers is None and cost_bps == 0:
        cached = _cache.load_backtest(factor, horizon_days, variant, freq, start_date, end_date)
        if cached is not None:
            logger.info(f'backtest cache hit: {factor} h={horizon_days} {variant}/{freq} {start_date}–{end_date}')
            return cached

    pd_freq = 'QE' if freq == 'Q' else 'YE'
    rebalance_dates = [
        ts.date()
        for ts in pd.date_range(start_date, end_date, freq=pd_freq)
    ]

    cross_sections: dict[datetime.date, pd.DataFrame] = {}
    dates_to_compute = []

    if tickers is None:
        for d in rebalance_dates:
            cached = _cache.load(d, variant)
            if cached is not None:
                cross_sections[d] = cached
            else:
                dates_to_compute.append(d)
    else:
        dates_to_compute = list(rebalance_dates)

    n_total = len(rebalance_dates)
    n_cached = n_total - len(dates_to_compute)
    logger.info(
        f'backtest {factor} h={horizon_days} {variant}/{freq} {start_date}–{end_date}: '
        f'{n_cached}/{n_total} cross-sections from cache, {len(dates_to_compute)} to compute'
    )

    if dates_to_compute:
        cross_sections.update(
            _compute_and_cache(
                dates_to_compute, variant, tickers,
                write_cache=(tickers is None),
            )
        )

    logger.info(f'backtest {factor}: computing forward returns via SQL...')
    fwd_returns = forward_returns_panel(rebalance_dates, horizon_days, tickers)
    logger.info(f'backtest {factor}: computing IC series and quintile returns...')
    result = compute_backtest(
        factor, cross_sections, fwd_returns,
        horizon_days=horizon_days, cost_bps=cost_bps,
    )
    if tickers is None and cost_bps == 0 and result['n_dates'] > 0:
        _cache.store_backtest(factor, horizon_days, variant, freq, start_date, end_date, result)
        logger.info(
            f'backtest {factor}: done — mean IC={result["mean_ic"]:.3f}, '
            f't-stat={result["ic_tstat"]:.2f}, ICIR={result.get("icir", float("nan")):.2f}, '
            f'n={result["n_dates"]} dates (result cached)'
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
    """Compute mean IC and ICIR across multiple forward-return horizons.

    Reuses cached cross-sections. Fetches raw fundamentals only for dates
    not in cache. Raw price data is always fetched (needed for all horizons).

    Returns
    -------
    DataFrame with columns: horizon, mean_ic, icir, n_dates.
    """
    pd_freq = 'QE' if freq == 'Q' else 'YE'
    rebalance_dates = [ts.date() for ts in pd.date_range(start_date, end_date, freq=pd_freq)]

    cross_sections: dict[datetime.date, pd.DataFrame] = {}
    dates_to_compute: list[datetime.date] = []

    if tickers is None:
        for d in rebalance_dates:
            cached = _cache.load(d, variant)
            if cached is not None:
                cross_sections[d] = cached
            else:
                dates_to_compute.append(d)
    else:
        dates_to_compute = list(rebalance_dates)

    if dates_to_compute:
        logger.info(f'factor decay {factor}: computing {len(dates_to_compute)} uncached cross-sections')
        cross_sections.update(
            _compute_and_cache(
                dates_to_compute, variant, tickers,
                write_cache=(tickers is None),
            )
        )

    if tickers is not None:
        cross_sections = {d: xs[xs.index.isin(tickers)] for d, xs in cross_sections.items()}

    rows = []
    for h in sorted(horizons):
        fwd = forward_returns_panel(rebalance_dates, h, tickers)
        res = compute_backtest(factor, cross_sections, fwd, horizon_days=h)
        rows.append({
            'horizon': h,
            'mean_ic': res['mean_ic'],
            'icir': res.get('icir', float('nan')),
            'n_dates': res['n_dates'],
        })
        logger.info(f'factor decay {factor} h={h}: mean_ic={res["mean_ic"]:.3f}, ICIR={res.get("icir", float("nan")):.2f}')

    return pd.DataFrame(rows)
