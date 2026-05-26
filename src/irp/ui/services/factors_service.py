"""Cross-section + ticker history retrieval for the UI layer.

Wraps `irp.factors.cross_section` + `ticker_factor_history` and bolts the
universe/sector/market/watchlist filtering on top so pages don't reach
into both `irp.factors` and `irp.query.simfin` separately.
"""
import datetime
from typing import Literal

import numpy as np
import pandas as pd

from irp.factors import cross_section as _cross_section
from irp.factors import ticker_factor_history as _ticker_factor_history
from irp.panel.load import load_prices_wide
from irp.ui.services import universe_service


def load_cross_section(
    as_of_date: datetime.date,
    variant: Literal['A', 'Q'] = 'A',
    market: str | None = None,
    sector: str | None = None,
    watchlist: str | None = None,
    enrich_company_columns: bool = True,
) -> pd.DataFrame:
    """Cached cross-section optionally filtered by market/sector/watchlist.

    When `enrich_company_columns` is True the result gains `Sector`,
    `Company Name`, and `Market` columns merged from the universe + SimFin
    companies tables. Tickers absent from the universe/companies merge are
    kept; only enrichment is best-effort.
    """
    tickers = universe_service.filter_tickers(market=market, sector=sector, watchlist=watchlist)
    xs = _cross_section(as_of_date, variant, tickers)
    if not enrich_company_columns or xs.empty:
        return xs

    u = universe_service.get_universe()[['Ticker', 'Market']]
    c = universe_service.get_companies()[['Ticker', 'Company Name', 'Sector']]
    meta = u.merge(c, on='Ticker', how='left').set_index('Ticker')
    return xs.join(meta, how='left')


def load_ticker_history(
    ticker: str, variant: Literal['A', 'Q'] = 'A',
) -> pd.DataFrame:
    """Factor history per filing date for one ticker."""
    return _ticker_factor_history(ticker, variant)


def compute_return_corr(
    tickers: list[str],
    window_days: int,
    as_of_date: datetime.date | None = None,
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Ticker × ticker log-return correlation from the price panel.

    Returns (corr_df, ticker_list, warning_or_None).
    corr_df is empty on error; warning is set on partial issues.
    """
    if as_of_date is None:
        as_of_date = datetime.date.today()

    panel = load_prices_wide('Close')
    as_of_np = np.datetime64(as_of_date, 'D')
    end_idx = int(np.searchsorted(panel.dates, as_of_np, side='right'))
    start_idx = max(0, end_idx - window_days)

    if end_idx - start_idx < 20:
        return pd.DataFrame(), [], 'Not enough price history in selected window'

    use_tickers = [t for t in tickers if t in panel.ticker_to_idx]
    if len(use_tickers) < 2:
        return pd.DataFrame(), [], 'Fewer than 2 tickers found in price panel'

    idxs = [panel.ticker_to_idx[t] for t in use_tickers]
    price_slice = panel.values[start_idx:end_idx, :][:, idxs]

    prices_df = pd.DataFrame(price_slice, columns=use_tickers)
    min_obs = int(price_slice.shape[0] * 0.8)
    prices_df = prices_df.dropna(axis=1, thresh=min_obs)
    if prices_df.shape[1] < 2:
        return pd.DataFrame(), [], 'Too few tickers with sufficient price history'

    log_ret = np.log(prices_df / prices_df.shift(1)).dropna(how='all')
    corr = log_ret.corr(method='pearson')

    warning = f'{len(prices_df.columns)} tickers — heatmap may be dense' if len(prices_df.columns) > 80 else None
    return corr, list(corr.columns), warning
