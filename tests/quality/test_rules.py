import pandas as pd
import pytest

from irp.quality.base import Finding, Registry, Severity
from irp.quality.rules.accounting import AccountingIdentity
from irp.quality.rules.impossible import ImpossibleValues
from irp.quality.rules.jumps import SuddenJumps
from irp.quality.rules.outliers import SectorOutliers
from irp.quality.runner import run


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
