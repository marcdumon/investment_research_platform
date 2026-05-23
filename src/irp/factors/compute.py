"""Cross-section factor computation orchestrator.

This is the only module in irp.factors that accesses the database.
Pure computation is delegated to valuation.py and profitability.py.
"""
import datetime
import logging
from typing import Literal

import pandas as pd

from irp.core.config import config
from irp.factors import cache as _cache
from irp.factors._cols import REPORT_DATE, TICKER
from irp.factors._pit import pit_latest, pit_prepare, pit_price, pit_ttm
from irp.factors.backtest import compute_backtest, compute_forward_returns
from irp.factors.valuation import compute_valuation
from irp.factors.profitability import compute_profitability
from irp.factors.momentum import compute_momentum
from irp.factors.leverage import compute_leverage
from irp.factors.growth import compute_growth
from irp.factors.piotroski import compute_piotroski
from irp.factors.registry import all_factors
from irp.query.simfin import fundamentals
from irp.query.yahoo import prices as yahoo_prices

logger = logging.getLogger(__name__)


def _cross_section_from_raw(
    raw_income: pd.DataFrame,
    raw_balance: pd.DataFrame,
    raw_cashflow: pd.DataFrame,
    raw_prices: pd.DataFrame,
    as_of_date: datetime.date,
    variant: Literal['A', 'Q'],
) -> pd.DataFrame:
    """PIT cross-section from pre-fetched raw DataFrames (no DB access)."""
    if variant == 'Q':
        income   = pit_ttm(raw_income,   as_of_date)
        cashflow = pit_ttm(raw_cashflow, as_of_date)
    else:
        income   = pit_latest(raw_income,   as_of_date)
        cashflow = pit_latest(raw_cashflow, as_of_date)
    balance = pit_latest(raw_balance,  as_of_date)
    prices  = pit_price(raw_prices,    as_of_date)

    if income.empty or balance.empty or cashflow.empty or prices.empty:
        return pd.DataFrame()

    val  = compute_valuation(income, balance, cashflow, prices)
    prof = compute_profitability(income, balance, cashflow)
    mom  = compute_momentum(raw_prices, as_of_date)
    lev  = compute_leverage(income, balance, cashflow)
    grow = compute_growth(raw_income, raw_cashflow, as_of_date, variant)
    piot = compute_piotroski(raw_income, raw_balance, raw_cashflow, as_of_date, variant)

    result = val.join(prof, how='outer').join(mom, how='outer').join(lev, how='outer').join(grow, how='outer').join(piot, how='outer')
    for f in all_factors():
        if f.positive_only and f.name in result.columns:
            result[f.name] = result[f.name].where(result[f.name] > 0)
    result.index.name = TICKER
    return result


def _compute_and_cache(
    dates: list[datetime.date],
    variant: str,
    raw_income: pd.DataFrame,
    raw_balance: pd.DataFrame,
    raw_cashflow: pd.DataFrame,
    raw_prices: pd.DataFrame,
    write_cache: bool = True,
) -> dict[datetime.date, pd.DataFrame]:
    """Compute cross-sections for `dates` in parallel; optionally write to cache.

    Returns dict of date → DataFrame (non-empty results only).
    Cache writes are serialised through the calling thread so no lock is needed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(d):
        return d, _cross_section_from_raw(
            raw_income, raw_balance, raw_cashflow, raw_prices, d, variant
        )

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
    DataFrame indexed by Ticker with valuation, profitability, and momentum columns.
    Tickers without sufficient data are silently absent.
    """
    if tickers is None:
        cached = _cache.load(as_of_date, variant)
        if cached is not None:
            return cached

    raw_income   = fundamentals(tickers, 'income',   variant)
    raw_balance  = fundamentals(tickers, 'balance',  variant)
    raw_cashflow = fundamentals(tickers, 'cashflow', variant)
    raw_prices   = yahoo_prices(tickers, end=as_of_date.isoformat())
    result = _cross_section_from_raw(
        raw_income, raw_balance, raw_cashflow, raw_prices, as_of_date, variant
    )
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

    raw_prices = pit_prepare(yahoo_prices(tickers), 'price')

    if dates_to_compute:
        logger.info(f'composite backtest: computing {len(dates_to_compute)} uncached cross-sections')
        raw_income   = pit_prepare(fundamentals(tickers, 'income',   variant), 'fundamental')
        raw_balance  = pit_prepare(fundamentals(tickers, 'balance',  variant), 'fundamental')
        raw_cashflow = pit_prepare(fundamentals(tickers, 'cashflow', variant), 'fundamental')
        cross_sections_raw.update(
            _compute_and_cache(
                dates_to_compute, variant,
                raw_income, raw_balance, raw_cashflow, raw_prices,
            )
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

    fwd_returns = compute_forward_returns(raw_prices, rebalance_dates, horizon_days)
    logger.info('composite backtest: computing IC series and quintile returns...')
    result = compute_backtest('__composite__', cross_sections_enriched, fwd_returns)
    logger.info(
        f'composite backtest: done — mean IC={result["mean_ic"]:.3f}, '
        f't-stat={result["ic_tstat"]:.2f}, n={result["n_dates"]} dates'
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
    if tickers is None:
        cached = _cache.load_backtest(factor, horizon_days, variant, freq, start_date, end_date)
        if cached is not None:
            logger.info(f'backtest cache hit: {factor} h={horizon_days} {variant}/{freq} {start_date}–{end_date}')
            return cached

    raw_income   = pit_prepare(fundamentals(tickers, 'income',   variant), 'fundamental')
    raw_balance  = pit_prepare(fundamentals(tickers, 'balance',  variant), 'fundamental')
    raw_cashflow = pit_prepare(fundamentals(tickers, 'cashflow', variant), 'fundamental')
    raw_prices   = pit_prepare(yahoo_prices(tickers),                      'price')

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

    cross_sections.update(
        _compute_and_cache(
            dates_to_compute, variant,
            raw_income, raw_balance, raw_cashflow, raw_prices,
            write_cache=(tickers is None),
        )
    )

    logger.info(f'backtest {factor}: computing IC series and quintile returns...')
    fwd_returns = compute_forward_returns(raw_prices, rebalance_dates, horizon_days)
    result = compute_backtest(factor, cross_sections, fwd_returns)
    if tickers is None and result['n_dates'] > 0:
        _cache.store_backtest(factor, horizon_days, variant, freq, start_date, end_date, result)
        logger.info(
            f'backtest {factor}: done — mean IC={result["mean_ic"]:.3f}, '
            f't-stat={result["ic_tstat"]:.2f}, n={result["n_dates"]} dates (result cached)'
        )
    return result
