import os

import pytest

from irp.sources.simfin import SimFinFundamentalsSource

pytestmark = pytest.mark.skipif(
    not os.environ.get("SIMFIN_API_KEY"),
    reason="SIMFIN_API_KEY not set",
)

# Columns guaranteed present in SimFin annual income statements
_EXPECTED_INCOME_COLS = {
    "Ticker",
    "Report Date",
    "Revenue",
    "Net Income",
    "Operating Income (Loss)",
}


def test_real_income_columns():
    ds = SimFinFundamentalsSource("income", variant="annual").fetch()
    missing = _EXPECTED_INCOME_COLS - set(ds.data.columns)
    assert not missing, f"Missing columns: {missing}"


def test_real_income_nonempty():
    ds = SimFinFundamentalsSource("income", variant="annual").fetch()
    assert len(ds.data) > 0


def test_real_income_schema_matches_data():
    ds = SimFinFundamentalsSource("income", variant="annual").fetch()
    assert set(ds.schema.keys()) == set(ds.data.columns)


def test_real_balance_columns():
    ds = SimFinFundamentalsSource("balance", variant="annual").fetch()
    assert "Total Assets" in ds.data.columns
    assert "Total Liabilities" in ds.data.columns


def test_real_cashflow_columns():
    ds = SimFinFundamentalsSource("cashflow", variant="annual").fetch()
    assert "Net Cash from Operating Activities" in ds.data.columns
