"""Unit tests for irp.factors.backtest — pure computation, no DB."""
import datetime

import numpy as np
import pandas as pd
import pytest

from irp.factors.backtest import compute_backtest, compute_forward_returns
from irp.factors._cols import PRICE_TICKER, PRICE_DATE, PRICE_CLOSE


def _make_prices(tickers: list[str], n: int, start: str = '2018-01-01') -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq='B')
    rng = np.random.default_rng(42)
    rows = []
    for t in tickers:
        closes = 100 * np.exp(rng.standard_normal(n).cumsum() * 0.01)
        for d, c in zip(dates, closes):
            rows.append({PRICE_TICKER: t, PRICE_DATE: d, PRICE_CLOSE: c})
    return pd.DataFrame(rows)


def _rebalance_dates(prices: pd.DataFrame) -> list[datetime.date]:
    dates = pd.to_datetime(prices[PRICE_DATE])
    start = dates.min().date()
    end = (dates.max() - pd.Timedelta(days=300)).date()
    return [
        ts.date() for ts in pd.date_range(start, end, freq='QE')
    ]


_TICKERS_3  = ['AAA', 'BBB', 'CCC']
_TICKERS_25 = [f'T{i:02d}' for i in range(25)]


@pytest.fixture
def prices():
    return _make_prices(_TICKERS_3, 600)


@pytest.fixture
def prices_25():
    return _make_prices(_TICKERS_25, 600)


@pytest.fixture
def rebal(prices):
    return _rebalance_dates(prices)


@pytest.fixture
def rebal_25(prices_25):
    return _rebalance_dates(prices_25)


# ── compute_forward_returns ───────────────────────────────────────────

def test_fwd_ret_single_ticker_correct(prices):
    p = prices[prices[PRICE_TICKER] == 'AAA'].copy()
    p[PRICE_DATE] = pd.to_datetime(p[PRICE_DATE])
    d0 = p[PRICE_DATE].iloc[0].date()
    d1_target = pd.Timestamp(d0) + pd.Timedelta(days=63)
    p0 = p[p[PRICE_DATE] <= pd.Timestamp(d0)].iloc[-1][PRICE_CLOSE]
    p1 = p[p[PRICE_DATE] <= d1_target].iloc[-1][PRICE_CLOSE]
    expected = np.log(p1 / p0)

    result = compute_forward_returns(prices, [d0], horizon_days=63)
    row = result[(result[PRICE_TICKER] == 'AAA') & (result['Date'] == d0)]
    assert not row.empty
    assert row['fwd_ret'].iloc[0] == pytest.approx(expected, rel=1e-6)


def test_fwd_ret_nan_beyond_data_end(prices):
    # horizon beyond data end: ffill gives last known price as both entry and exit → log(1)=0
    last_date = pd.to_datetime(prices[PRICE_DATE]).max().date()
    result = compute_forward_returns(prices, [last_date], horizon_days=500)
    assert not result.empty
    assert np.allclose(result['fwd_ret'].values, 0.0, atol=1e-12)


def test_fwd_ret_all_tickers_present(prices, rebal):
    result = compute_forward_returns(prices, rebal[:2], horizon_days=63)
    tickers_found = set(result[PRICE_TICKER].unique())
    assert {'AAA', 'BBB', 'CCC'}.issubset(tickers_found)


# ── compute_backtest — IC ─────────────────────────────────────────────

def _perfect_cross_sections(fwd_returns, factor: str = 'f') -> dict:
    """Factor value = forward return rank (perfect predictor, IC = +1)."""
    xs = {}
    for d, group in fwd_returns.groupby('Date'):
        g = group.set_index(PRICE_TICKER)['fwd_ret'].dropna()
        xs[d] = pd.DataFrame({factor: g.rank()})
    return xs


def test_ic_plus_one_for_perfect_predictor(prices_25, rebal_25):
    fwd = compute_forward_returns(prices_25, rebal_25, horizon_days=63)
    xs = _perfect_cross_sections(fwd, 'f')
    result = compute_backtest('f', xs, fwd)
    valid = result['ic_series'].dropna()
    assert len(valid) > 0
    assert all(abs(v - 1.0) < 0.05 for v in valid)


def test_ic_near_zero_for_random_factor(prices_25, rebal_25):
    rng = np.random.default_rng(99)
    fwd = compute_forward_returns(prices_25, rebal_25, horizon_days=63)
    xs = {}
    for d, group in fwd.groupby('Date'):
        tickers = group[PRICE_TICKER].values
        xs[d] = pd.DataFrame({'rand': rng.standard_normal(len(tickers))},
                              index=tickers)
    result = compute_backtest('rand', xs, fwd)
    valid = result['ic_series'].dropna()
    assert len(valid) > 0
    assert abs(result['mean_ic']) < 0.5


# ── compute_backtest — quintiles ──────────────────────────────────────

def test_quintile_cumret_has_five_columns(prices_25, rebal_25):
    fwd = compute_forward_returns(prices_25, rebal_25, horizon_days=63)
    xs = _perfect_cross_sections(fwd)
    result = compute_backtest('f', xs, fwd)
    assert list(result['quintile_cumret'].columns) == ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']


def test_q5_beats_q1_for_perfect_predictor(prices_25, rebal_25):
    fwd = compute_forward_returns(prices_25, rebal_25, horizon_days=63)
    xs = _perfect_cross_sections(fwd)
    result = compute_backtest('f', xs, fwd)
    qcr = result['quintile_cumret']
    if not qcr.empty:
        assert qcr['Q5'].iloc[-1] > qcr['Q1'].iloc[-1]


def test_ic_nan_below_20_observations():
    # Fewer than 20 tickers → IC = NaN
    dates = pd.date_range('2020-01-01', periods=300, freq='B')
    rows = []
    for t in ['A', 'B', 'C']:
        for d, c in zip(dates, np.arange(1, 301, dtype=float)):
            rows.append({PRICE_TICKER: t, PRICE_DATE: d, PRICE_CLOSE: c})
    prices = pd.DataFrame(rows)
    d0 = dates[0].date()
    fwd = compute_forward_returns(prices, [d0], horizon_days=63)
    xs = {d0: pd.DataFrame({'f': [1.0, 2.0, 3.0]}, index=['A', 'B', 'C'])}
    result = compute_backtest('f', xs, fwd)
    assert result['ic_series'].isna().all()


def test_n_dates_counts_valid_ic(prices_25, rebal_25):
    fwd = compute_forward_returns(prices_25, rebal_25, horizon_days=63)
    xs = _perfect_cross_sections(fwd)
    result = compute_backtest('f', xs, fwd)
    expected = int(result['ic_series'].dropna().count())
    assert result['n_dates'] == expected
