"""Tests for irp.factors.leverage.compute_leverage."""
import pandas as pd
import pytest

from irp.factors._cols import (
    TICKER,
    OPERATING_INCOME,
    INTEREST_EXPENSE,
    TOTAL_EQUITY,
    CASH_AND_ST_INVESTMENTS,
    SHORT_TERM_DEBT,
    LONG_TERM_DEBT,
    DA,
    CFO,
)
from irp.factors.leverage import compute_leverage


def _income(**kw) -> pd.DataFrame:
    base = {TICKER: 'AAPL', OPERATING_INCOME: 100, INTEREST_EXPENSE: -10}
    return pd.DataFrame([{**base, **kw}])


def _balance(**kw) -> pd.DataFrame:
    base = {
        TICKER: 'AAPL',
        TOTAL_EQUITY: 200,
        CASH_AND_ST_INVESTMENTS: 50,
        SHORT_TERM_DEBT: 30,
        LONG_TERM_DEBT: 70,
    }
    return pd.DataFrame([{**base, **kw}])


def _cashflow(**kw) -> pd.DataFrame:
    base = {TICKER: 'AAPL', DA: 20}
    return pd.DataFrame([{**base, **kw}])


class TestDebtEquity:
    def test_basic(self):
        result = compute_leverage(_income(), _balance(), _cashflow())
        # (30 + 70) / 200 = 0.5
        assert result.loc['AAPL', 'debt_equity'] == pytest.approx(0.5)

    def test_zero_equity_gives_na(self):
        result = compute_leverage(_income(), _balance(**{TOTAL_EQUITY: 0}), _cashflow())
        assert pd.isna(result.loc['AAPL', 'debt_equity'])

    def test_negative_equity_kept(self):
        result = compute_leverage(_income(), _balance(**{TOTAL_EQUITY: -50}), _cashflow())
        # (30 + 70) / -50 = -2.0
        assert result.loc['AAPL', 'debt_equity'] == pytest.approx(-2.0)

    def test_null_debt_treated_as_zero(self):
        result = compute_leverage(
            _income(),
            _balance(**{SHORT_TERM_DEBT: None, LONG_TERM_DEBT: None}),
            _cashflow(),
        )
        assert result.loc['AAPL', 'debt_equity'] == pytest.approx(0.0)


class TestNetDebtEbitda:
    def test_basic(self):
        # net_debt = 30 + 70 - 50 = 50; ebitda = 100 + 20 = 120
        result = compute_leverage(_income(), _balance(), _cashflow())
        assert result.loc['AAPL', 'net_debt_ebitda'] == pytest.approx(50 / 120)

    def test_negative_ebitda_gives_negative_ratio(self):
        result = compute_leverage(
            _income(**{OPERATING_INCOME: -200}),
            _balance(),
            _cashflow(**{DA: 10}),
        )
        # ebitda = -200 + 10 = -190; negative ratio kept
        assert result.loc['AAPL', 'net_debt_ebitda'] < 0

    def test_zero_ebitda_gives_na(self):
        result = compute_leverage(
            _income(**{OPERATING_INCOME: 0}),
            _balance(),
            _cashflow(**{DA: 0}),
        )
        assert pd.isna(result.loc['AAPL', 'net_debt_ebitda'])


class TestInterestCoverage:
    def test_basic(self):
        # ebit = 100; interest_paid = abs(-10) = 10; coverage = 10
        result = compute_leverage(_income(), _balance(), _cashflow())
        assert result.loc['AAPL', 'interest_coverage'] == pytest.approx(10.0)

    def test_positive_interest_expense_also_works(self):
        # Some sources may record it as positive
        result = compute_leverage(
            _income(**{INTEREST_EXPENSE: 10}),
            _balance(),
            _cashflow(),
        )
        assert result.loc['AAPL', 'interest_coverage'] == pytest.approx(10.0)

    def test_zero_interest_gives_na(self):
        result = compute_leverage(
            _income(**{INTEREST_EXPENSE: 0}),
            _balance(),
            _cashflow(),
        )
        assert pd.isna(result.loc['AAPL', 'interest_coverage'])

    def test_no_interest_expense_column(self):
        inc = _income().drop(columns=[INTEREST_EXPENSE])
        result = compute_leverage(inc, _balance(), _cashflow())
        assert pd.isna(result.loc['AAPL', 'interest_coverage'])


class TestOutputStructure:
    def test_indexed_by_ticker(self):
        result = compute_leverage(_income(), _balance(), _cashflow())
        assert result.index.name == TICKER
        assert 'AAPL' in result.index

    def test_has_all_columns(self):
        result = compute_leverage(_income(), _balance(), _cashflow())
        assert {'debt_equity', 'net_debt_ebitda', 'interest_coverage'}.issubset(result.columns)

    def test_multi_ticker(self):
        inc = pd.DataFrame([
            {TICKER: 'AAPL', OPERATING_INCOME: 100, INTEREST_EXPENSE: -10},
            {TICKER: 'MSFT', OPERATING_INCOME: 200, INTEREST_EXPENSE: -20},
        ])
        bal = pd.DataFrame([
            {TICKER: 'AAPL', TOTAL_EQUITY: 200, CASH_AND_ST_INVESTMENTS: 50, SHORT_TERM_DEBT: 30, LONG_TERM_DEBT: 70},
            {TICKER: 'MSFT', TOTAL_EQUITY: 400, CASH_AND_ST_INVESTMENTS: 100, SHORT_TERM_DEBT: 60, LONG_TERM_DEBT: 140},
        ])
        cf = pd.DataFrame([
            {TICKER: 'AAPL', DA: 20},
            {TICKER: 'MSFT', DA: 40},
        ])
        result = compute_leverage(inc, bal, cf)
        assert len(result) == 2
        assert set(result.index) == {'AAPL', 'MSFT'}

    def test_ticker_absent_from_any_input_dropped(self):
        inc = _income()
        bal = _balance(**{TICKER: 'MSFT'})
        cf = _cashflow()
        result = compute_leverage(inc, bal, cf)
        assert result.empty
