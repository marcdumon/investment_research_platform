"""Row accessors for SimFin tables (`income`, `balance`, `cashflow`, `companies`)."""
from typing import Literal

import pandas as pd

from irp.data._common import db, ticker_filter


def fundamentals(
    tickers: str | list[str] | None = None,
    statement: Literal['income', 'balance', 'cashflow'] = 'income',
    variant: Literal['A', 'Q'] = 'A',
) -> pd.DataFrame:
    """Financial statement data. variant: 'A' annual, 'Q' quarterly."""
    filters, params = ['Period = ?'], [variant]
    clause, vals = ticker_filter(tickers)
    if clause:
        filters.append(clause)
        params.extend(vals)
    where = f'WHERE {" AND ".join(filters)}'
    return (
        db()
        .execute(
            f'SELECT * FROM {statement} {where} ORDER BY Ticker, "Fiscal Year" DESC',
            params,
        )
        .df()
    )


_FP_ORDER = {'FY': 0, 'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
_META = {
    'Ticker', 'SrcId', 'Src', 'Currency', 'Fiscal Year', 'Fiscal Period',
    'Report Date', 'Publish Date', 'Restated Date', 'Market', 'Period',
}


def _parse_period(p: str) -> tuple[int, str, Literal['A', 'Q']]:
    """'2024FY' -> (2024,'FY','A'); '2024Q2' -> (2024,'Q2','Q')."""
    if p.endswith('FY'):
        return int(p[:-2]), 'FY', 'A'
    return int(p[:4]), p[4:], 'Q'


def statement(
    ticker: str,
    name: Literal['income', 'balance', 'cashflow'] = 'income',
    periods: list[str] | None = None,
    items: list[str] | None = None,
) -> pd.DataFrame:
    """Pull a financial statement for one ticker, one or more periods.

    Returns a tidy view with line items as rows and periods as columns
    (most-recent first). `periods` are strings like '2024FY' or '2024Q2'.
    `items` optionally filters which line items to include.
    """
    if periods is None:
        df_a = fundamentals(ticker, name, 'A')
        df_q = fundamentals(ticker, name, 'Q')
        df = pd.concat([df_a, df_q], ignore_index=True)
    else:
        parsed = [_parse_period(p) for p in periods]
        variants = {p[2] for p in parsed}
        parts = [fundamentals(ticker, name, v) for v in variants]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    # Balance stores annual rows with Fiscal Period='Q4'. Force 'FY' when Period='A'.
    df = df.copy()
    df.loc[df['Period'] == 'A', 'Fiscal Period'] = 'FY'

    if periods is not None:
        keys = {(fy, fp, v) for fy, fp, v in parsed}
        mask = [
            (int(fy), str(fp), str(p)) in keys
            for fy, fp, p in zip(df['Fiscal Year'], df['Fiscal Period'], df['Period'])
        ]
        df = df[mask]
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df['_period'] = [
        f'{int(fy)}FY' if p == 'A' else f'{int(fy)}{fp}'
        for fy, fp, p in zip(df['Fiscal Year'], df['Fiscal Period'], df['Period'])
    ]
    df['_sort'] = [
        (-int(fy), _FP_ORDER.get(str(fp), 99))
        for fy, fp in zip(df['Fiscal Year'], df['Fiscal Period'])
    ]
    df = df.sort_values('_sort').drop(columns=['_sort'])
    df = df.drop_duplicates('_period', keep='first')

    item_cols = [c for c in df.columns if c not in _META and c != '_period']
    if items is not None:
        item_cols = [c for c in item_cols if c in items]
    return df.set_index('_period')[item_cols].T


def companies(tickers: str | list[str] | None = None) -> pd.DataFrame:
    """Company metadata: name, sector, industry, market, ISIN."""
    clause, params = ticker_filter(tickers)
    where = f'WHERE {clause}' if clause else ''
    return (
        db().execute(f'SELECT * FROM companies {where} ORDER BY Ticker', params).df()
    )


def universe(
    sector: str | None = None,
    industry: str | None = None,
    market: str | None = None,
) -> list[str]:
    """Return tickers matching sector/industry/market filters."""
    filters, params = [], []
    if sector is not None:
        filters.append('Sector = ?')
        params.append(sector)
    if industry is not None:
        filters.append('Industry = ?')
        params.append(industry)
    if market is not None:
        filters.append('Market = ?')
        params.append(market)
    where = f'WHERE {" AND ".join(filters)}' if filters else ''
    rows = (
        db()
        .execute(
            f'SELECT DISTINCT Ticker FROM companies {where} ORDER BY Ticker', params
        )
        .fetchall()
    )
    return [r[0] for r in rows]
