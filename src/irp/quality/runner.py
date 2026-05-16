from typing import Literal

import duckdb
import pandas as pd

from irp.core.config import config
from irp.data import fundamentals
from irp.quality.edgar import filing_url
from irp.quality.rules import REGISTRY

_KEY = ['Ticker', 'Fiscal Year', 'Fiscal Period', 'Period']


def _cik_map(tickers: list[str]) -> dict[str, int]:
    if not tickers:
        return {}
    placeholders = ','.join(['?'] * len(tickers))
    with duckdb.connect(str(config.database.path), read_only=True) as con:
        rows = con.execute(
            f'SELECT Ticker, CIK FROM companies WHERE Ticker IN ({placeholders}) AND CIK IS NOT NULL',
            tickers,
        ).fetchall()
    return {t: int(c) for t, c in rows}


def run(tickers: list[str] | None = None, variant: Literal['A', 'Q'] = 'A') -> pd.DataFrame:
    """Run all registered quality rules. Returns violations enriched with CIK + EDGAR link."""
    data = {
        'income':   fundamentals(tickers, 'income',   variant),
        'balance':  fundamentals(tickers, 'balance',  variant),
        'cashflow': fundamentals(tickers, 'cashflow', variant),
    }
    findings = []
    for rule in REGISTRY:
        v = rule.fn(data)
        if len(v):
            v = v.copy()
            v.insert(0, 'Rule', rule.name)
            v.insert(1, 'Statement', rule.statement)
            findings.append(v)
    if not findings:
        return pd.DataFrame()
    df = pd.concat(findings, ignore_index=True)

    # Attach Report Date (from income — most reliable cross-statement)
    rd = data['income'][_KEY + ['Report Date']].drop_duplicates(_KEY)
    df = df.merge(rd, on=_KEY, how='left')

    # Attach CIK + EDGAR URL
    cik = _cik_map(df['Ticker'].unique().tolist())
    df['CIK'] = df['Ticker'].map(cik)
    df['EDGAR'] = [
        filing_url(
            int(c) if pd.notna(c) else None,
            rdate.strftime('%Y-%m-%d') if pd.notna(rdate) else None,
            str(p),
        )
        for c, rdate, p in zip(df['CIK'], df['Report Date'], df['Period'])
    ]
    return df
