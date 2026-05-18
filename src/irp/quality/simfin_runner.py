from typing import Literal

import pandas as pd

from irp.core.db import db
from irp.data.simfin import fundamentals
from irp.quality.edgar import filing_url
from irp.quality.simfin_reviews import load_reviews, period_str
from irp.quality.simfin_rules import REGISTRY

_KEY = ['Ticker', 'Fiscal Year', 'Fiscal Period', 'Period']


def _cik_map(tickers: list[str]) -> dict[str, int]:
    if not tickers:
        return {}
    placeholders = ','.join(['?'] * len(tickers))
    rows = db().execute(
        f'SELECT Ticker, CIK FROM companies WHERE Ticker IN ({placeholders}) AND CIK IS NOT NULL',
        tickers,
    ).fetchall()
    return {t: int(c) for t, c in rows}


def run(
    tickers: list[str] | None = None,
    variant: Literal['A', 'Q'] = 'A',
    skip_reviewed: bool = True,
) -> pd.DataFrame:
    """Run all registered quality rules. Returns violations enriched with
    Period_str, CIK, EDGAR. When skip_reviewed=True, filters out items
    already in the anomaly_reviews.toml."""
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

    df['Period_str'] = [
        period_str(int(fy), str(fp), str(p))
        for fy, fp, p in zip(df['Fiscal Year'], df['Fiscal Period'], df['Period'])
    ]

    if skip_reviewed:
        reviewed = load_reviews()
        mask = [
            (t, ps, r) not in reviewed
            for t, ps, r in zip(df['Ticker'], df['Period_str'], df['Rule'])
        ]
        df = df[mask].reset_index(drop=True)
        if df.empty:
            return df

    rd = data['income'][_KEY + ['Report Date']].drop_duplicates(_KEY)
    df = df.merge(rd, on=_KEY, how='left')

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

    df = df.sort_values(
        ['Ticker', 'Fiscal Year', 'Fiscal Period', 'Period', 'Rule'],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)
    return df
