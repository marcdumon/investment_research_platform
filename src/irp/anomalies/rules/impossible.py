from collections.abc import Callable
from typing import ClassVar

import pandas as pd

from irp.anomalies.base import Finding, Rule, Severity, register

# (table, column, predicate, label, exclude_fn)
# exclude_fn(df) -> boolean Series; matching rows are excluded from findings.
# Use None for no exclusion.
_Check = tuple[
    str,
    str,
    Callable[[pd.Series], pd.Series],
    str,
    Callable[[pd.DataFrame], pd.Series] | None,
]

_DEFAULT_CHECKS: list[_Check] = [
    # Q4 Revenue is excluded: Q4 = Annual - Q1 - Q2 - Q3 is verified by
    # the quarterly_consistency rule instead (unverifiable via 10-Q filing).
    (
        "income",
        "Revenue",
        lambda s: s < 0,
        "Revenue is negative",
        lambda df: df["Fiscal Period"].eq("Q4"),
    ),
    ("income", "Shares (Diluted)", lambda s: s <= 0, "Diluted shares non-positive", None),
    ("income", "Shares (Basic)", lambda s: s <= 0, "Basic shares non-positive", None),
    ("balance", "Total Assets", lambda s: s <= 0, "Total Assets non-positive", None),
]


@register
class ImpossibleValues(Rule):
    name = "impossible_value"
    description = "Values that are physically or accounting-wise impossible"
    severity = Severity.ERROR

    checks: ClassVar[list[_Check]] = _DEFAULT_CHECKS

    def check(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for table, column, predicate, label, exclude_fn in self.checks:
            if table not in data or column not in data[table].columns:
                continue
            df = data[table].dropna(subset=[column])
            mask = predicate(df[column])
            if exclude_fn is not None and "Fiscal Period" in df.columns:
                mask = mask & ~exclude_fn(df)
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
