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


# Income statement flow items (additive across fiscal quarters: Q1+Q2+Q3+Q4 = Annual)
_FLOW_CHECKS: list[tuple[str, str]] = [
    ("income", "Revenue"),
]


@register
class QuarterlyConsistency(Rule):
    """Annual = Q1 + Q2 + Q3 + Q4 for income statement flow columns.

    Uses SimFin's Fiscal Period column so fiscal-year-end date doesn't matter.
    Flags the Annual row when the sum of quarters diverges beyond tolerance.
    """

    name = "quarterly_consistency"
    description = "Annual = Q1 + Q2 + Q3 + Q4 for income statement flow columns"
    severity = Severity.ERROR
    tolerance: float = 0.01

    def check(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for table_name, col in _FLOW_CHECKS:
            tbl = data.get(table_name)
            if tbl is None or col not in tbl.columns:
                continue
            result = self._check_column(tbl, table_name, col)
            if not result.empty:
                frames.append(result)
        return pd.concat(frames, ignore_index=True) if frames else Finding.empty_df()

    def _check_column(self, tbl: pd.DataFrame, table_name: str, col: str) -> pd.DataFrame:
        needed = ["Ticker", "Fiscal Year", "Fiscal Period", "variant", "Report Date", col]
        df = tbl[[c for c in needed if c in tbl.columns]].copy().dropna(subset=[col])
        df["report_date"] = df["Report Date"].astype(str).str[:10]

        annual = (
            df[df["variant"] == "A"]
            .rename(columns={col: "annual"})
            [["Ticker", "Fiscal Year", "annual", "report_date"]]
        )

        qtrs = df[df["variant"] == "Q"].copy()
        qtrs["qtr"] = qtrs["Fiscal Period"]  # "Q1"…"Q4" — fiscal, not calendar

        wide = (
            qtrs.pivot_table(
                index=["Ticker", "Fiscal Year"],
                columns="qtr",
                values=col,
                aggfunc="first",
            )
            .reset_index()
        )
        for q in ("Q1", "Q2", "Q3", "Q4"):
            if q not in wide.columns:
                wide[q] = float("nan")

        merged = wide.merge(annual, on=["Ticker", "Fiscal Year"], how="inner")
        merged = merged.dropna(subset=["Q1", "Q2", "Q3", "Q4", "annual"])

        merged["sum_qtrs"] = merged["Q1"] + merged["Q2"] + merged["Q3"] + merged["Q4"]
        denom = merged["annual"].abs().replace(0, float("nan"))
        merged["rel_err"] = (merged["annual"] - merged["sum_qtrs"]).abs() / denom

        bad = merged[merged["rel_err"] > self.tolerance].copy()
        if bad.empty:
            return Finding.empty_df()

        annual_str  = bad["annual"].round(0).astype("Int64").astype(str)
        sum_str     = bad["sum_qtrs"].round(0).astype("Int64").astype(str)
        pct_str     = (bad["rel_err"] * 100).round(1).astype(str)

        return pd.DataFrame(
            {
                "table": table_name,
                "ticker": bad["Ticker"].values,
                "variant": "A",
                "report_date": bad["report_date"].fillna("").values,
                "column": col,
                "value": bad["rel_err"].round(4).values,
                "rule": self.name,
                "detail": (
                    "Annual=" + annual_str
                    + ", Q1+Q2+Q3+Q4=" + sum_str
                    + ", rel_err=" + pct_str + "%"
                ).values,
                "severity": self.severity,
            }
        )
