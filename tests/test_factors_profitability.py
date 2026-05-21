"""Tests for irp.factors.profitability.compute_profitability."""
import pandas as pd
import pytest

from irp.factors._cols import (
    CASH_AND_ST_INVESTMENTS,
    CFO,
    GROSS_PROFIT,
    LONG_TERM_DEBT,
    NET_INCOME,
    OPERATING_INCOME,
    REVENUE,
    SHORT_TERM_DEBT,
    TICKER,
    TOTAL_ASSETS,
    TOTAL_EQUITY,
)
from irp.factors.profitability import compute_profitability

# ── Fixture builders ──────────────────────────────────────────────────

def _income(**kw) -> pd.DataFrame:
    base = {
        TICKER: 'AAPL',
        REVENUE: 100_000,
        GROSS_PROFIT: 40_000,
        OPERATING_INCOME: 15_000,
        NET_INCOME: 10_000,
    }
    return pd.DataFrame([{**base, **kw}])


def _balance(**kw) -> pd.DataFrame:
    base = {
        TICKER: 'AAPL',
        TOTAL_ASSETS: 120_000,
        TOTAL_EQUITY: 50_000,
        SHORT_TERM_DEBT: 5_000,
        LONG_TERM_DEBT: 20_000,
        CASH_AND_ST_INVESTMENTS: 10_000,
    }
    return pd.DataFrame([{**base, **kw}])


def _cashflow(**kw) -> pd.DataFrame:
    base = {TICKER: 'AAPL', CFO: 12_000}
    return pd.DataFrame([{**base, **kw}])


# ── Tests ─────────────────────────────────────────────────────────────

class TestComputeProfitability:
    def test_output_indexed_by_ticker(self):
        result = compute_profitability(_income(), _balance(), _cashflow())
        assert result.index.name == TICKER
        assert 'AAPL' in result.index

    def test_gross_margin(self):
        result = compute_profitability(_income(), _balance(), _cashflow())
        assert result.loc['AAPL', 'gross_margin'] == pytest.approx(0.40)

    def test_op_margin(self):
        result = compute_profitability(_income(), _balance(), _cashflow())
        assert result.loc['AAPL', 'op_margin'] == pytest.approx(0.15)

    def test_net_margin(self):
        result = compute_profitability(_income(), _balance(), _cashflow())
        assert result.loc['AAPL', 'net_margin'] == pytest.approx(0.10)

    def test_roe(self):
        # roe = 10_000 / 50_000 = 0.20
        result = compute_profitability(_income(), _balance(), _cashflow())
        assert result.loc['AAPL', 'roe'] == pytest.approx(0.20)

    def test_roa(self):
        result = compute_profitability(_income(), _balance(), _cashflow())
        assert result.loc['AAPL', 'roa'] == pytest.approx(10_000 / 120_000)

    def test_fcf_margin(self):
        # fcf_margin = 12_000 / 100_000 = 0.12
        result = compute_profitability(_income(), _balance(), _cashflow())
        assert result.loc['AAPL', 'fcf_margin'] == pytest.approx(0.12)

    def test_roic(self):
        # net_debt = 5000 + 20000 - 10000 = 15_000
        # ic = 50_000 + 15_000 = 65_000
        # roic = 15_000 / 65_000
        result = compute_profitability(_income(), _balance(), _cashflow())
        assert result.loc['AAPL', 'roic'] == pytest.approx(15_000 / 65_000)

    def test_margins_na_when_revenue_zero(self):
        result = compute_profitability(
            _income(**{REVENUE: 0}), _balance(), _cashflow()
        )
        for col in ('gross_margin', 'op_margin', 'net_margin', 'fcf_margin'):
            assert pd.isna(result.loc['AAPL', col]), f'{col} should be NA'

    def test_roe_negative_when_equity_negative(self):
        result = compute_profitability(
            _income(), _balance(**{TOTAL_EQUITY: -5_000}), _cashflow()
        )
        assert result.loc['AAPL', 'roe'] < 0

    def test_roa_negative_when_net_income_negative(self):
        result = compute_profitability(
            _income(**{NET_INCOME: -2_000}), _balance(), _cashflow()
        )
        assert result.loc['AAPL', 'roa'] < 0

    def test_expected_columns_present(self):
        result = compute_profitability(_income(), _balance(), _cashflow())
        expected = {'gross_margin', 'op_margin', 'net_margin', 'roe', 'roa', 'roic', 'fcf_margin'}
        assert expected.issubset(set(result.columns))

    def test_ticker_absent_when_missing_from_balance(self):
        empty_bal = pd.DataFrame(
            columns=[TICKER, TOTAL_ASSETS, TOTAL_EQUITY,
                     SHORT_TERM_DEBT, LONG_TERM_DEBT, CASH_AND_ST_INVESTMENTS]
        )
        result = compute_profitability(_income(), empty_bal, _cashflow())
        assert result.empty
