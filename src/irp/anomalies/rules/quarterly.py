import pandas as pd

from irp.anomalies.base import Finding, Rule, Severity, register

# (table, column) pairs that are income-statement flow items (additive across fiscal quarters)
_FLOW_CHECKS: list[tuple[str, str]] = [
    ("income", "Revenue"),
]


@register
class QuarterlyConsistency(Rule):
    """Check that Q4 = Annual - Q1 - Q2 - Q3 for income statement flow columns.

    Uses SimFin's `Fiscal Period` column ("Q1"…"Q4") so fiscal-year-end date
    doesn't matter — Apple's fiscal Q4 (July-Sep) is handled the same as a
    calendar-year company's Q4 (Oct-Dec).
    """

    name = "quarterly_consistency"
    description = "Q4 = Annual - Q1 - Q2 - Q3 for income statement flow columns"
    severity = Severity.ERROR
    tolerance: float = 0.01  # relative error threshold (1%)

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

    def _check_column(
        self, tbl: pd.DataFrame, table_name: str, col: str
    ) -> pd.DataFrame:
        needed = ["Ticker", "Fiscal Year", "Fiscal Period", "variant", "Report Date", col]
        df = tbl[[c for c in needed if c in tbl.columns]].copy().dropna(subset=[col])
        df["report_date"] = df["Report Date"].astype(str).str[:10]

        # ── annual rows ───────────────────────────────────────────────────────
        annual = (
            df[df["variant"] == "A"]
            .rename(columns={col: "annual"})
            [["Ticker", "Fiscal Year", "annual"]]
        )

        # ── quarterly rows — pivot to one column per fiscal quarter ───────────
        qtrs = df[df["variant"] == "Q"].copy()
        qtrs["qtr"] = qtrs["Fiscal Period"]  # already "Q1"…"Q4" — fiscal, not calendar

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

        # ── merge and compute derivation error ────────────────────────────────
        merged = wide.merge(annual, on=["Ticker", "Fiscal Year"], how="inner")
        merged = merged.dropna(subset=["Q1", "Q2", "Q3", "Q4", "annual"])

        merged["derived"] = merged["annual"] - merged["Q1"] - merged["Q2"] - merged["Q3"]
        denom = merged["annual"].abs().replace(0, float("nan"))
        merged["rel_err"] = (merged["Q4"] - merged["derived"]).abs() / denom

        bad = merged[merged["rel_err"] > self.tolerance].copy()
        if bad.empty:
            return Finding.empty_df()

        # re-join to recover Q4 report_date
        q4_meta = (
            qtrs[qtrs["qtr"] == "Q4"][["Ticker", "Fiscal Year", "report_date"]]
            .drop_duplicates(["Ticker", "Fiscal Year"])
        )
        bad = bad.merge(q4_meta, on=["Ticker", "Fiscal Year"], how="left")

        derived_str = bad["derived"].round(0).astype("Int64").astype(str)
        reported_str = bad["Q4"].round(0).astype("Int64").astype(str)
        pct_str = (bad["rel_err"] * 100).round(1).astype(str)

        return pd.DataFrame(
            {
                "table": table_name,
                "ticker": bad["Ticker"].values,
                "variant": "Q",
                "report_date": bad["report_date"].fillna("").values,
                "column": col,
                "value": bad["rel_err"].round(4).values,
                "rule": self.name,
                "detail": (
                    "derived Q4=" + derived_str
                    + ", reported=" + reported_str
                    + " (" + pct_str + "% discrepancy)"
                ).values,
                "severity": self.severity,
            }
        )
