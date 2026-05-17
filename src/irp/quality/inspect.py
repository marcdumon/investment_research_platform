import pandas as pd

from irp.data import fundamentals
from irp.quality.rules import REGISTRY

_BASE = ['Ticker', 'Fiscal Year', 'Fiscal Period', 'Period', 'Report Date']


def inspect(rule_name: str, ticker: str, fy: int, fp: str, period: str) -> pd.DataFrame:
    """Return rule-relevant SimFin columns for a (ticker, fy, fp, period).
    For cross-statement rules, returns rows from each statement with a _stmt label.
    """
    rule = next((r for r in REGISTRY if r.name == rule_name), None)
    if rule is None:
        return pd.DataFrame()
    extra = list(rule.columns)

    if rule.statement == 'cross':
        rows = []
        for stmt in ('income', 'balance', 'cashflow'):
            df = fundamentals([ticker], stmt, period)
            df = df[(df['Fiscal Year'] == fy) & (df['Fiscal Period'] == fp)]
            if df.empty:
                continue
            cols = [c for c in _BASE + extra if c in df.columns]
            rows.append(df[cols].assign(_stmt=stmt))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    df = fundamentals([ticker], rule.statement, period)
    df = df[(df['Fiscal Year'] == fy) & (df['Fiscal Period'] == fp)]
    cols = [c for c in _BASE + extra if c in df.columns]
    return df[cols].reset_index(drop=True)
