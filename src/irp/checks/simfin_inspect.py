from typing import Literal, cast

import pandas as pd

from irp.checks.simfin_rules import REGISTRY
from irp.query.simfin import SimFinStatement, fundamentals

_BASE = ['Ticker', 'Fiscal Year', 'Fiscal Period', 'Period', 'Report Date']


def _inspect(rule_name: str, ticker: str, fy: int, fp: str, period: Literal['A','Q']) -> pd.DataFrame:
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
            df = fundamentals([ticker], stmt, period)  # type: ignore[arg-type]
            df = df[(df['Fiscal Year'] == fy) & (df['Fiscal Period'] == fp)]
            if df.empty:
                continue
            cols = [c for c in _BASE + extra if c in df.columns]
            rows.append(df[cols].assign(_stmt=stmt))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    df = fundamentals([ticker], cast(SimFinStatement, rule.statement), period)
    df = df[(df['Fiscal Year'] == fy) & (df['Fiscal Period'] == fp)]
    cols = [c for c in _BASE + extra if c in df.columns]
    return df[cols].reset_index(drop=True)
