from typing import ClassVar

import pandas as pd

from irp.anomalies.base import Finding, Rule, Severity, register


@register
class AccountingIdentity(Rule):
    name = "accounting_identity"
    description = "Total Assets = Total Liabilities + Total Equity (within tolerance)"
    severity = Severity.ERROR

    tolerance: ClassVar[float] = 0.01  # 1% relative error

    def check(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if "balance" not in data:
            return Finding.empty_df()

        cols = ["Total Assets", "Total Liabilities", "Total Equity"]
        df = data["balance"].dropna(subset=cols)
        if df.empty:
            return Finding.empty_df()

        residual = df["Total Assets"] - df["Total Liabilities"] - df["Total Equity"]
        rel_err = residual.abs() / df["Total Assets"].abs().replace(0.0, float("nan"))
        mask = rel_err > self.tolerance
        bad = df[mask].copy()
        if bad.empty:
            return Finding.empty_df()

        bad_rel = rel_err[mask]
        detail = (
            "Assets="
            + bad["Total Assets"].round(0).astype("int64").astype(str)
            + ", Liab+Eq="
            + (bad["Total Liabilities"] + bad["Total Equity"])
            .round(0)
            .astype("int64")
            .astype(str)
            + ", rel_err="
            + (bad_rel * 100).round(2).astype(str)
            + "%"
        )

        return pd.DataFrame(
            {
                "table": "balance",
                "ticker": bad["Ticker"].values,
                "variant": bad["variant"].values,
                "report_date": bad["Report Date"].astype(str).str[:10].values,
                "column": "Total Assets, Total Liabilities, Total Equity",
                "value": bad_rel.round(6).values,
                "rule": self.name,
                "detail": detail.values,
                "severity": self.severity,
            }
        )
