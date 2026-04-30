from typing import ClassVar

import pandas as pd

from irp.quality.base import Finding, Rule, Severity, register

# (table, column) pairs to check.  Extend to add new metrics.
_Check = tuple[str, str]

_DEFAULT_CHECKS: list[_Check] = [
    ("income", "Revenue"),
    ("income", "Net Income"),
    ("balance", "Total Assets"),
]


@register
class SuddenJumps(Rule):
    name = "sudden_jump"
    description = "Period-over-period change exceeds configured thresholds"
    severity = Severity.WARNING

    up_threshold: ClassVar[float] = 5.0  # +500%
    down_threshold: ClassVar[float] = -0.80  # -80%
    checks: ClassVar[list[_Check]] = _DEFAULT_CHECKS

    def check(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for table, column in self.checks:
            if table not in data or column not in data[table].columns:
                continue

            df = (
                data[table]
                .dropna(subset=[column])
                .sort_values(["Ticker", "variant", "Report Date"])
                .copy()
            )
            df["_prev"] = df.groupby(["Ticker", "variant"])[column].shift(1)
            df["_pct"] = (df[column] - df["_prev"]) / df["_prev"].abs().replace(
                0.0, float("nan")
            )

            mask = (df["_pct"] > self.up_threshold) | (df["_pct"] < self.down_threshold)
            bad = df[mask].dropna(subset=["_pct"])
            if bad.empty:
                continue

            detail = (
                column
                + ": "
                + bad["_prev"].round(0).astype("int64").astype(str)
                + " → "
                + bad[column].round(0).astype("int64").astype(str)
                + " ("
                + (bad["_pct"] * 100).round(1).astype(str)
                + "%)"
            )

            frames.append(
                pd.DataFrame(
                    {
                        "table": table,
                        "ticker": bad["Ticker"].values,
                        "variant": bad["variant"].values,
                        "report_date": bad["Report Date"].astype(str).str[:10].values,
                        "column": column,
                        "value": bad["_pct"].round(4).values,
                        "rule": self.name,
                        "detail": detail.values,
                        "severity": self.severity,
                    }
                )
            )

        return pd.concat(frames, ignore_index=True) if frames else Finding.empty_df()
