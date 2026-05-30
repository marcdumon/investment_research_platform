"""Stooq anomaly runner. Mirrors irp.checks.simfin_runner."""
import pandas as pd

from irp.core.db import db
from irp.checks.stooq_reviews import _load_reviews as load_reviews
from irp.checks.stooq_rules import REGISTRY


def _run(skip_reviewed: bool = True) -> pd.DataFrame:
    """Run all registered stooq rules. Returns aggregated findings sorted
    by `count` descending. When `skip_reviewed=True`, drops entries already
    recorded in `data/data_quality/stooq_anomaly_reviews.toml`.

    Columns: Rule, Ticker, Year, count, sample_dates, Period_str, Market.
    `Market` is a ' / '-joined list of all markets the ticker belongs to
    (some tickers exist in multiple markets, e.g. stocks + crypto).
    """
    con = db()
    findings = []
    for rule in REGISTRY:
        df = rule.fn(con)
        if len(df):
            df = df.copy()
            df.insert(0, 'Rule', rule.name)
            findings.append(df)
    if not findings:
        return pd.DataFrame()
    df = pd.concat(findings, ignore_index=True)
    df['Period_str'] = df['Year'].astype(int).astype(str)

    market_rows = con.execute("""
        SELECT Ticker, STRING_AGG(DISTINCT Market, ' / ') AS Market
        FROM universe
        GROUP BY Ticker
    """).fetchall()
    market_map = {t: m for t, m in market_rows}
    df['Market'] = df['Ticker'].map(market_map).fillna('')

    if skip_reviewed:
        reviewed = load_reviews()
        if reviewed:
            mask = [
                (t, p, r) not in reviewed
                for t, p, r in zip(df['Ticker'], df['Period_str'], df['Rule'])
            ]
            df = df[mask]
    return df.sort_values('count', ascending=False).reset_index(drop=True)
