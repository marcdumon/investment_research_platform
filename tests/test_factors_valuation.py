"""Tests for irp.factors.valuation.compute_valuation."""
import pandas as pd
import pytest

from irp.factors._cols import (
    CASH_AND_ST_INVESTMENTS,
    CFI,
    CFO,
    DA,
    LONG_TERM_DEBT,
    NET_INCOME,
    OPERATING_INCOME,
    PRICE_CLOSE,
    PRICE_TICKER,
    REVENUE,
    SHARES_DILUTED_BAL,
    SHORT_TERM_DEBT,
    TICKER,
    TOTAL_EQUITY,
)
from irp.factors.valuation import compute_valuation

# ── Fixture builders ──────────────────────────────────────────────────

def _income(**kw) -> pd.DataFrame:
    base = {
        TICKER: 'AAPL',
        NET_INCOME: 10_000,
        REVENUE: 100_000,
        OPERATING_INCOME: 15_000,
    }
    return pd.DataFrame([{**base, **kw}])


def _balance(**kw) -> pd.DataFrame:
    base = {
        TICKER: 'AAPL',
        TOTAL_EQUITY: 50_000,
        CASH_AND_ST_INVESTMENTS: 10_000,
        SHORT_TERM_DEBT: 5_000,
        LONG_TERM_DEBT: 20_000,
        SHARES_DILUTED_BAL: 1_000,
    }
    return pd.DataFrame([{**base, **kw}])


def _cashflow(**kw) -> pd.DataFrame:
    base = {TICKER: 'AAPL', CFO: 12_000, CFI: -3_000, DA: 2_000}
    return pd.DataFrame([{**base, **kw}])


def _prices(**kw) -> pd.DataFrame:
    base = {PRICE_TICKER: 'AAPL', PRICE_CLOSE: 150.0}
    return pd.DataFrame([{**base, **kw}])


# ── Tests ─────────────────────────────────────────────────────────────

class TestComputeValuation:
    def test_output_indexed_by_ticker(self):
        result = compute_valuation(_income(), _balance(), _cashflow(), _prices())
        assert result.index.name == TICKER
        assert 'AAPL' in result.index

    def test_mktcap(self):
        # mktcap = 150 * 1000 = 150_000
        result = compute_valuation(_income(), _balance(), _cashflow(), _prices())
        assert result.loc['AAPL', 'mktcap'] == pytest.approx(150_000)

    def test_pe(self):
        # pe = 150_000 / 10_000 = 15.0
        result = compute_valuation(_income(), _balance(), _cashflow(), _prices())
        assert result.loc['AAPL', 'pe'] == pytest.approx(15.0)

    def test_pb(self):
        # pb = 150_000 / 50_000 = 3.0
        result = compute_valuation(_income(), _balance(), _cashflow(), _prices())
        assert result.loc['AAPL', 'pb'] == pytest.approx(3.0)

    def test_ps(self):
        # ps = 150_000 / 100_000 = 1.5
        result = compute_valuation(_income(), _balance(), _cashflow(), _prices())
        assert result.loc['AAPL', 'ps'] == pytest.approx(1.5)

    def test_pe_is_na_when_net_income_zero(self):
        result = compute_valuation(
            _income(**{NET_INCOME: 0}), _balance(), _cashflow(), _prices()
        )
        assert pd.isna(result.loc['AAPL', 'pe'])

    def test_pb_is_negative_when_equity_negative(self):
        result = compute_valuation(
            _income(), _balance(**{TOTAL_EQUITY: -10_000}), _cashflow(), _prices()
        )
        assert result.loc['AAPL', 'pb'] < 0

    def test_ev_ebitda(self):
        # net_debt = 5000 + 20000 - 10000 = 15_000
        # ev = 150_000 + 15_000 = 165_000
        # ebitda = 15_000 + 2_000 = 17_000
        # ev_ebitda = 165_000 / 17_000
        result = compute_valuation(_income(), _balance(), _cashflow(), _prices())
        expected = 165_000 / 17_000
        assert result.loc['AAPL', 'ev_ebitda'] == pytest.approx(expected)

    def test_fcf_yield(self):
        # fcf = 12_000 + (-3_000) = 9_000; fcf_yield = 9_000 / 150_000 = 0.06
        result = compute_valuation(_income(), _balance(), _cashflow(), _prices())
        assert result.loc['AAPL', 'fcf_yield'] == pytest.approx(0.06)

    def test_ticker_absent_when_prices_empty(self):
        empty_prices = pd.DataFrame(columns=[PRICE_TICKER, PRICE_CLOSE])
        result = compute_valuation(_income(), _balance(), _cashflow(), empty_prices)
        assert result.empty

    def test_expected_columns_present(self):
        result = compute_valuation(_income(), _balance(), _cashflow(), _prices())
        expected = {'mktcap', 'pe', 'pb', 'ps', 'ev_ebitda', 'ev_ebit', 'ev_sales', 'fcf_yield'}
        assert expected.issubset(set(result.columns))

    def test_multi_ticker(self):
        inc = pd.concat([_income(), _income(**{TICKER: 'MSFT'})], ignore_index=True)
        bal = pd.concat([_balance(), _balance(**{TICKER: 'MSFT'})], ignore_index=True)
        cf  = pd.concat([_cashflow(), _cashflow(**{TICKER: 'MSFT'})], ignore_index=True)
        px  = pd.concat([_prices(), _prices(**{PRICE_TICKER: 'MSFT'})], ignore_index=True)
        result = compute_valuation(inc, bal, cf, px)
        assert set(result.index) == {'AAPL', 'MSFT'}

    def test_da_null_uses_zero_for_ebitda(self):
        # DA = None → ebitda = ebit + 0 = operating_income
        result = compute_valuation(
            _income(), _balance(), _cashflow(**{DA: None}), _prices()
        )
        # ebitda = 15_000; ev_ebitda = 165_000 / 15_000 = 11.0
        assert result.loc['AAPL', 'ev_ebitda'] == pytest.approx(165_000 / 15_000)
