"""Cross-section + ticker history retrieval for the UI layer.

Wraps `irp.factors.cross_section` + `ticker_factor_history` and bolts the
universe/sector/market/watchlist filtering on top so pages don't reach
into both `irp.factors` and `irp.query.simfin` separately.
"""
import datetime
from typing import Literal

import pandas as pd

from irp.factors import cross_section as _cross_section
from irp.factors import ticker_factor_history as _ticker_factor_history
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
