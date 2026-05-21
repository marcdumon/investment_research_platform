"""Tests for point-in-time alignment in irp.factors._pit."""
import datetime

import pandas as pd
import pytest

from irp.factors._cols import (
    PRICE_CLOSE,
    PRICE_DATE,
    PRICE_TICKER,
    REPORT_DATE,
    TICKER,
)
from irp.factors._pit import pit_latest, pit_price


def _fund(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _prices(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestPitLatest:
    def test_returns_most_recent_row_per_ticker(self):
        df = _fund(
            {TICKER: 'AAPL', REPORT_DATE: '2024-02-15', 'Revenue': 100},
            {TICKER: 'AAPL', REPORT_DATE: '2023-08-10', 'Revenue': 90},
            {TICKER: 'MSFT', REPORT_DATE: '2024-01-20', 'Revenue': 200},
        )
        result = pit_latest(df, datetime.date(2024, 6, 1))
        assert len(result) == 2
        aapl = result.loc[result[TICKER] == 'AAPL'].iloc[0]
        assert aapl['Revenue'] == 100

    def test_excludes_rows_after_as_of_date(self):
        df = _fund(
            {TICKER: 'AAPL', REPORT_DATE: '2024-07-01', 'Revenue': 120},
            {TICKER: 'AAPL', REPORT_DATE: '2024-02-15', 'Revenue': 100},
        )
        result = pit_latest(df, datetime.date(2024, 6, 1))
        assert len(result) == 1
        assert result.iloc[0]['Revenue'] == 100

    def test_inclusive_on_exact_as_of_date(self):
        df = _fund({TICKER: 'AAPL', REPORT_DATE: '2024-06-01', 'Revenue': 100})
        result = pit_latest(df, datetime.date(2024, 6, 1))
        assert len(result) == 1

    def test_returns_empty_when_all_rows_after_as_of(self):
        df = _fund({TICKER: 'AAPL', REPORT_DATE: '2025-01-01', 'Revenue': 100})
        result = pit_latest(df, datetime.date(2024, 6, 1))
        assert result.empty

    def test_tickers_with_only_future_rows_are_absent(self):
        df = _fund(
            {TICKER: 'AAPL', REPORT_DATE: '2024-02-01', 'Revenue': 100},
            {TICKER: 'GOOG', REPORT_DATE: '2025-01-01', 'Revenue': 999},
        )
        result = pit_latest(df, datetime.date(2024, 6, 1))
        assert 'GOOG' not in result[TICKER].values
        assert 'AAPL' in result[TICKER].values

    def test_does_not_mutate_input(self):
        df = _fund({TICKER: 'AAPL', REPORT_DATE: '2024-02-01', 'Revenue': 100})
        original_dtype = df[REPORT_DATE].dtype
        pit_latest(df, datetime.date(2024, 6, 1))
        assert df[REPORT_DATE].dtype == original_dtype

    def test_preserves_all_columns(self):
        df = _fund({TICKER: 'AAPL', REPORT_DATE: '2024-02-01', 'Revenue': 100, 'NetIncome': 20})
        result = pit_latest(df, datetime.date(2024, 6, 1))
        assert set(df.columns) == set(result.columns)


class TestPitPrice:
    def test_returns_closest_price_on_or_before_as_of(self):
        df = _prices(
            {PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-05-30', PRICE_CLOSE: 180.0},
            {PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-05-31', PRICE_CLOSE: 182.0},
            {PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-06-03', PRICE_CLOSE: 190.0},
        )
        result = pit_price(df, datetime.date(2024, 6, 1))
        assert result.iloc[0][PRICE_CLOSE] == pytest.approx(182.0)

    def test_inclusive_on_exact_as_of_date(self):
        df = _prices({PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-06-01', PRICE_CLOSE: 180.0})
        result = pit_price(df, datetime.date(2024, 6, 1))
        assert len(result) == 1

    def test_returns_empty_when_all_prices_after_as_of(self):
        df = _prices({PRICE_TICKER: 'AAPL', PRICE_DATE: '2025-01-01', PRICE_CLOSE: 200.0})
        result = pit_price(df, datetime.date(2024, 6, 1))
        assert result.empty

    def test_one_row_per_ticker(self):
        df = _prices(
            {PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-05-01', PRICE_CLOSE: 170.0},
            {PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-05-30', PRICE_CLOSE: 180.0},
            {PRICE_TICKER: 'MSFT', PRICE_DATE: '2024-05-30', PRICE_CLOSE: 400.0},
        )
        result = pit_price(df, datetime.date(2024, 6, 1))
        assert len(result) == 2

    def test_output_contains_required_columns(self):
        df = _prices({PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-05-30', PRICE_CLOSE: 180.0})
        result = pit_price(df, datetime.date(2024, 6, 1))
        assert {PRICE_TICKER, PRICE_DATE, PRICE_CLOSE}.issubset(set(result.columns))

    def test_does_not_mutate_input(self):
        df = _prices({PRICE_TICKER: 'AAPL', PRICE_DATE: '2024-05-30', PRICE_CLOSE: 180.0})
        original_dtype = df[PRICE_DATE].dtype
        pit_price(df, datetime.date(2024, 6, 1))
        assert df[PRICE_DATE].dtype == original_dtype
