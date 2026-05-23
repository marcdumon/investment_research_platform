"""Tests for irp.factors.growth.compute_growth."""
import datetime

import pandas as pd
import pytest

from irp.factors._cols import NET_INCOME, PUBLISH_DATE, REPORT_DATE, REVENUE, TICKER
from irp.factors.growth import compute_growth

_AS_OF = datetime.date(2023, 12, 31)
_PRIOR = datetime.date(2022, 12, 31)


def _income_row(ticker, report_date, revenue, net_income) -> dict:
    return {
        TICKER: ticker,
        REPORT_DATE: report_date,
        PUBLISH_DATE: None,
        REVENUE: revenue,
        NET_INCOME: net_income,
        'Gross Profit': 0,
        'Operating Income (Loss)': 0,
        'Interest Expense, Net': 0,
    }


def _raw_income(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _cashflow_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=[TICKER, REPORT_DATE, 'Net Cash from Operating Activities'])


class TestComputeGrowth:
    def _make_income(self):
        return _raw_income(
            _income_row('AAPL', '2023-09-30', 380_000, 100_000),
            _income_row('AAPL', '2022-09-30', 300_000,  80_000),
            _income_row('MSFT', '2023-06-30', 200_000,  60_000),
            _income_row('MSFT', '2022-06-30', 160_000,  50_000),
        )

    def test_output_indexed_by_ticker(self):
        df = compute_growth(self._make_income(), _cashflow_empty(), _AS_OF, 'A')
        assert df.index.name == TICKER
        assert set(df.columns) >= {'rev_growth_1y', 'earn_growth_1y'}

    def test_revenue_growth(self):
        df = compute_growth(self._make_income(), _cashflow_empty(), _AS_OF, 'A')
        # AAPL: (380k - 300k) / 300k ≈ 0.2667
        assert df.loc['AAPL', 'rev_growth_1y'] == pytest.approx(80_000 / 300_000)

    def test_earnings_growth(self):
        df = compute_growth(self._make_income(), _cashflow_empty(), _AS_OF, 'A')
        # AAPL: (100k - 80k) / 80k = 0.25
        assert df.loc['AAPL', 'earn_growth_1y'] == pytest.approx(0.25)

    def test_no_prior_year_returns_empty(self):
        # Only current-year rows — no prior data
        raw = _raw_income(_income_row('AAPL', '2023-09-30', 380_000, 100_000))
        df = compute_growth(raw, _cashflow_empty(), _AS_OF, 'A')
        assert df.empty or df['rev_growth_1y'].isna().all()

    def test_negative_revenue_base_uses_abs(self):
        raw = _raw_income(
            _income_row('LOSS', '2023-09-30', 50_000, 0),
            _income_row('LOSS', '2022-09-30', -20_000, 0),
        )
        df = compute_growth(raw, _cashflow_empty(), _AS_OF, 'A')
        # (50k - (-20k)) / |-20k| = 70k / 20k = 3.5
        assert df.loc['LOSS', 'rev_growth_1y'] == pytest.approx(3.5)

    def test_ticker_missing_prior_drops_from_output(self):
        raw = _raw_income(
            _income_row('AAPL', '2023-09-30', 380_000, 100_000),
            _income_row('AAPL', '2022-09-30', 300_000,  80_000),
            _income_row('NEW',  '2023-09-30', 100_000,  10_000),  # no prior
        )
        df = compute_growth(raw, _cashflow_empty(), _AS_OF, 'A')
        assert 'AAPL' in df.index
        assert 'NEW' not in df.index
