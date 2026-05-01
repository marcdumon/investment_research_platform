import pandas as pd
import pytest

from irp.anomalies.base import Finding, Registry, Severity
from irp.anomalies.rules.accounting import AccountingIdentity
from irp.anomalies.rules.impossible import ImpossibleValues
from irp.anomalies.rules.jumps import SuddenJumps
from irp.anomalies.rules.outliers import SectorOutliers
from irp.anomalies.rules.accounting import QuarterlyConsistency
from irp.anomalies.runner import run


def _balance(**overrides) -> pd.DataFrame:
    row = {
        "Ticker": "AAA", "variant": "A", "Report Date": "2023-12-31",
        "Total Assets": 1000.0, "Total Liabilities": 600.0, "Total Equity": 400.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _income(**overrides) -> pd.DataFrame:
    row = {
        "Ticker": "AAA", "variant": "A", "Report Date": "2023-12-31",
        "Revenue": 500.0, "Net Income": 50.0, "Operating Income (Loss)": 70.0,
        "Shares (Diluted)": 100.0, "Shares (Basic)": 100.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# AccountingIdentity
# ---------------------------------------------------------------------------

class TestAccountingIdentity:
    rule = AccountingIdentity()

    def test_balanced_row_no_findings(self):
        assert self.rule.check({"balance": _balance()}).empty

    def test_imbalance_above_threshold_flagged(self):
        df = _balance()
        df["Total Equity"] = 350.0   # residual = 50 / 1000 = 5% > 1%
        result = self.rule.check({"balance": df})
        assert len(result) == 1
        assert result.iloc[0]["severity"] == Severity.ERROR
        assert result.iloc[0]["rule"] == "accounting_identity"

    def test_within_tolerance_no_finding(self):
        df = _balance()
        df["Total Equity"] = 395.0   # residual = 5 / 1000 = 0.5% < 1%
        assert self.rule.check({"balance": df}).empty

    def test_missing_table_returns_empty(self):
        assert self.rule.check({}).empty

    def test_custom_tolerance_via_config(self):
        strict = AccountingIdentity(tolerance=0.001)
        df = _balance()
        df["Total Equity"] = 995.0   # residual = 5 / 1000 = 0.5% > 0.1%
        assert not strict.check({"balance": df}).empty

    def test_result_has_finding_schema(self):
        df = _balance()
        df["Total Equity"] = 0.0
        result = self.rule.check({"balance": df})
        assert list(result.columns) == Finding.columns()


# ---------------------------------------------------------------------------
# ImpossibleValues
# ---------------------------------------------------------------------------

class TestImpossibleValues:
    rule = ImpossibleValues()

    def test_clean_data_no_findings(self):
        assert self.rule.check({"income": _income(), "balance": _balance()}).empty

    def test_negative_revenue_flagged(self):
        df = _income()
        df["Revenue"] = -100.0
        result = self.rule.check({"income": df})
        assert (result["column"] == "Revenue").any()
        assert (result["severity"] == Severity.ERROR).all()

    def test_zero_shares_diluted_flagged(self):
        df = _income()
        df["Shares (Diluted)"] = 0.0
        result = self.rule.check({"income": df})
        assert (result["column"] == "Shares (Diluted)").any()

    def test_nonpositive_total_assets_flagged(self):
        df = _balance()
        df["Total Assets"] = -500.0
        result = self.rule.check({"balance": df})
        assert (result["column"] == "Total Assets").any()

    def test_missing_column_skipped_gracefully(self):
        df = _income().drop(columns=["Revenue"])
        result = self.rule.check({"income": df})
        assert not (result["column"] == "Revenue").any() if not result.empty else True

    def test_negative_revenue_q4_not_flagged(self):
        # Q4 Revenue is excluded — caught by quarterly_consistency instead
        df = _income(Revenue=-100.0, variant="Q", **{"Fiscal Period": "Q4"})
        result = self.rule.check({"income": df})
        assert result.empty or not (result["column"] == "Revenue").any()

    def test_negative_revenue_q1_still_flagged(self):
        df = _income(Revenue=-100.0, variant="Q", **{"Fiscal Period": "Q1"})
        result = self.rule.check({"income": df})
        assert (result["column"] == "Revenue").any()


# ---------------------------------------------------------------------------
# SuddenJumps
# ---------------------------------------------------------------------------

class TestSuddenJumps:
    rule = SuddenJumps()

    def _two_periods(self, r1: float, r2: float) -> pd.DataFrame:
        return pd.DataFrame([
            {"Ticker": "AAA", "variant": "A", "Report Date": "2022-12-31",
             "Revenue": r1, "Net Income": 10.0},
            {"Ticker": "AAA", "variant": "A", "Report Date": "2023-12-31",
             "Revenue": r2, "Net Income": 10.0},
        ])

    def test_no_jump_no_finding(self):
        assert self.rule.check({"income": self._two_periods(100.0, 120.0)}).empty

    def test_large_jump_up_flagged(self):
        result = self.rule.check({"income": self._two_periods(100.0, 700.0)})
        assert (result["column"] == "Revenue").any()

    def test_large_drop_flagged(self):
        result = self.rule.check({"income": self._two_periods(1000.0, 100.0)})
        assert (result["column"] == "Revenue").any()

    def test_first_period_not_flagged(self):
        df = self._two_periods(100.0, 700.0).iloc[:1]
        assert self.rule.check({"income": df}).empty

    def test_custom_thresholds_via_config(self):
        strict = SuddenJumps(up_threshold=0.1, down_threshold=-0.1)
        result = strict.check({"income": self._two_periods(100.0, 115.0)})
        assert (result["column"] == "Revenue").any()


# ---------------------------------------------------------------------------
# SectorOutliers
# ---------------------------------------------------------------------------

class TestSectorOutliers:
    rule = SectorOutliers()

    def _build(self, revenues: list[float], outlier: float) -> dict[str, pd.DataFrame]:
        n = len(revenues) + 1
        tickers = [f"T{i:02d}" for i in range(n)]
        income = pd.DataFrame({
            "Ticker": tickers, "variant": ["A"] * n,
            "Report Date": ["2023-12-31"] * n,
            "Revenue": revenues + [outlier],
            "Net Income": [10.0] * n,
            "Operating Income (Loss)": [5.0] * n,
        })
        companies  = pd.DataFrame({"Ticker": tickers, "IndustryId": [1] * n})
        industries = pd.DataFrame({"IndustryId": [1], "Sector": ["Technology"]})
        return {"income": income, "companies": companies, "industries": industries}

    def test_extreme_outlier_flagged(self):
        data = self._build([100.0, 105.0, 95.0, 102.0, 98.0], outlier=1_000_000.0)
        assert not self.rule.check(data).empty

    def test_inlier_not_flagged(self):
        data = self._build([100.0, 105.0, 95.0, 102.0, 98.0], outlier=110.0)
        result = self.rule.check(data)
        assert result.empty or not (result["column"] == "Revenue").any()

    def test_too_few_peers_skipped(self):
        data = self._build([100.0, 105.0], outlier=1_000_000.0)  # 3 total < min_peers=5
        assert self.rule.check(data).empty

    def test_missing_tables_returns_empty(self):
        assert self.rule.check({"income": _income()}).empty

    def test_custom_fence_via_config(self):
        tight = SectorOutliers(iqr_fence=0.5)
        data  = self._build([100.0, 105.0, 95.0, 102.0, 98.0], outlier=130.0)
        assert not tight.check(data).empty


# ---------------------------------------------------------------------------
# QuarterlyConsistency
# ---------------------------------------------------------------------------

def _quarterly_income(
    ticker: str = "AAA",
    fiscal_year: int = 2023,
    annual: float = 400.0,
    q1: float = 100.0,
    q2: float = 100.0,
    q3: float = 100.0,
    q4: float = 100.0,
) -> pd.DataFrame:
    """Build an income DataFrame with one annual row + four quarterly rows."""
    rows = [
        {
            "Ticker": ticker, "Fiscal Year": fiscal_year,
            "Fiscal Period": "FY", "variant": "A",
            "Report Date": f"{fiscal_year}-12-31", "Revenue": annual,
        },
        {
            "Ticker": ticker, "Fiscal Year": fiscal_year,
            "Fiscal Period": "Q1", "variant": "Q",
            "Report Date": f"{fiscal_year}-03-31", "Revenue": q1,
        },
        {
            "Ticker": ticker, "Fiscal Year": fiscal_year,
            "Fiscal Period": "Q2", "variant": "Q",
            "Report Date": f"{fiscal_year}-06-30", "Revenue": q2,
        },
        {
            "Ticker": ticker, "Fiscal Year": fiscal_year,
            "Fiscal Period": "Q3", "variant": "Q",
            "Report Date": f"{fiscal_year}-09-30", "Revenue": q3,
        },
        {
            "Ticker": ticker, "Fiscal Year": fiscal_year,
            "Fiscal Period": "Q4", "variant": "Q",
            "Report Date": f"{fiscal_year}-12-31", "Revenue": q4,
        },
    ]
    return pd.DataFrame(rows)


class TestQuarterlyConsistency:
    rule = QuarterlyConsistency()

    def test_consistent_q4_no_finding(self):
        df = _quarterly_income(annual=400, q1=100, q2=100, q3=100, q4=100)
        assert self.rule.check({"income": df}).empty

    def test_mismatched_q4_flagged(self):
        # derived Q4 = 400 - 100 - 100 - 100 = 100, reported = 5 → 95% error
        df = _quarterly_income(annual=400, q1=100, q2=100, q3=100, q4=5)
        result = self.rule.check({"income": df})
        assert len(result) == 1
        assert result.iloc[0]["rule"] == "quarterly_consistency"
        assert result.iloc[0]["column"] == "Revenue"
        assert result.iloc[0]["value"] > 0.01

    def test_within_tolerance_no_finding(self):
        # 0.5% discrepancy < 1% tolerance
        df = _quarterly_income(annual=400, q1=100, q2=100, q3=100, q4=99.5)
        assert self.rule.check({"income": df}).empty

    def test_missing_quarter_skipped(self):
        # only Q1-Q3 and Annual — no Q4 row
        df = _quarterly_income(annual=400, q1=100, q2=100, q3=100, q4=100)
        df = df[df["Fiscal Period"] != "Q4"]
        assert self.rule.check({"income": df}).empty

    def test_missing_table_returns_empty(self):
        assert self.rule.check({}).empty

    def test_result_has_finding_schema(self):
        df = _quarterly_income(annual=400, q1=100, q2=100, q3=100, q4=5)
        result = self.rule.check({"income": df})
        assert list(result.columns) == Finding.columns()

    def test_custom_tolerance(self):
        strict = QuarterlyConsistency(tolerance=0.001)
        # 0.5% discrepancy → flagged with strict tolerance
        df = _quarterly_income(annual=400, q1=100, q2=100, q3=100, q4=99.5)
        assert not strict.check({"income": df}).empty

    def test_non_december_fiscal_year(self):
        # Apple-like: fiscal year ends September, Q4 = Jul-Sep
        rows = [
            {"Ticker": "AAPL", "Fiscal Year": 2023, "Fiscal Period": "FY",
             "variant": "A", "Report Date": "2023-09-30", "Revenue": 400.0},
            {"Ticker": "AAPL", "Fiscal Year": 2023, "Fiscal Period": "Q1",
             "variant": "Q", "Report Date": "2022-12-31", "Revenue": 100.0},
            {"Ticker": "AAPL", "Fiscal Year": 2023, "Fiscal Period": "Q2",
             "variant": "Q", "Report Date": "2023-03-31", "Revenue": 100.0},
            {"Ticker": "AAPL", "Fiscal Year": 2023, "Fiscal Period": "Q3",
             "variant": "Q", "Report Date": "2023-06-30", "Revenue": 100.0},
            {"Ticker": "AAPL", "Fiscal Year": 2023, "Fiscal Period": "Q4",
             "variant": "Q", "Report Date": "2023-09-30", "Revenue": 50.0},  # wrong
        ]
        df = pd.DataFrame(rows)
        result = self.rule.check({"income": df})
        assert len(result) == 1  # flagged correctly despite non-Dec fiscal year


# ---------------------------------------------------------------------------
# run() — dependency injection, isolated registry
# ---------------------------------------------------------------------------

class TestRunnerDI:
    def test_run_with_explicit_rules(self):
        df = _balance()
        df["Total Equity"] = 0.0
        result = run({"balance": df}, rules=[AccountingIdentity()])
        assert not result.empty
        assert (result["rule"] == "accounting_identity").all()

    def test_run_empty_rules_returns_empty_df(self):
        result = run({"income": _income()}, rules=[])
        assert result.empty
        assert list(result.columns) == Finding.columns()

    def test_registry_deduplication(self):
        reg = Registry()
        reg.register(AccountingIdentity)
        reg.register(AccountingIdentity)  # duplicate — should be ignored
        assert len(reg.rules()) == 1

    def test_registry_config_override(self):
        reg = Registry()
        reg.register(AccountingIdentity, tolerance=0.001)
        rule = reg.rules()[0]
        assert rule.tolerance == 0.001
