from collections.abc import Callable
from typing import ClassVar

import pandas as pd

from irp.anomalies.base import Finding, Rule, Severity, register

# Extend this list to add new impossible-value checks — no other changes needed.
_Check = tuple[str, str, Callable[[pd.Series], pd.Series], str]

_DEFAULT_CHECKS: list[_Check] = [
    ("income", "Revenue", lambda s: s < 0, "Revenue is negative"),
    ("income", "Shares (Diluted)", lambda s: s <= 0, "Diluted shares non-positive"),
    ("income", "Shares (Basic)", lambda s: s <= 0, "Basic shares non-positive"),
    ("balance", "Total Assets", lambda s: s <= 0, "Total Assets non-positive"),
]


@register
class ImpossibleValues(Rule):
    name = "impossible_value"
    description = "Values that are physically or accounting-wise impossible"
    severity = Severity.ERROR

    checks: ClassVar[list[_Check]] = _DEFAULT_CHECKS

    def check(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for table, column, predicate, label in self.checks:
            if table not in data or column not in data[table].columns:
                continue
            df = data[table].dropna(subset=[column])
            mask = predicate(df[column])
            bad = df[mask]
            if bad.empty:
                continue

            frames.append(
                pd.DataFrame(
                    {
                        "table": table,
                        "ticker": bad["Ticker"].values,
                        "variant": bad["variant"].values,
                        "report_date": bad["Report Date"].astype(str).str[:10].values,
                        "column": column,
                        "value": bad[column].astype(float).values,
                        "rule": self.name,
                        "detail": label,
                        "severity": self.severity,
                    }
                )
            )

        return pd.concat(frames, ignore_index=True) if frames else Finding.empty_df()
