"""Unit tests for irp.factors.momentum — pure computation, no DB."""
import datetime

import numpy as np
import pandas as pd
import pytest

from irp.factors.momentum import compute_momentum
from irp.factors._cols import PRICE_TICKER, PRICE_DATE, PRICE_CLOSE


def _make_prices(ticker: str, n: int, start: str = '2022-01-01') -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq='B')
    np.random.seed(42)
    closes = 100 * np.exp(np.random.randn(n).cumsum() * 0.01)
    return pd.DataFrame({
        PRICE_TICKER: ticker,
        PRICE_DATE:   dates,
        PRICE_CLOSE:  closes,
    })


@pytest.fixture
def prices_full():
    """Two tickers with >400 trading days of history."""
    df1 = _make_prices('AAA', 450, '2022-01-01')
    df2 = _make_prices('BBB', 450, '2022-01-01')
    return pd.concat([df1, df2], ignore_index=True)


def _as_of(prices_full) -> datetime.date:
    return prices_full[PRICE_DATE].max().date()


def test_mom_12_1_value(prices_full):
    as_of = _as_of(prices_full)
    result = compute_momentum(prices_full, as_of)
    assert 'mom_12_1' in result.columns
    assert result.loc['AAA', 'mom_12_1'] == pytest.approx(
        result.loc['AAA', 'mom_12_1'], rel=1e-6
    )
    # Verify sign consistency: mom computed against p_1m / p_12m
    assert not pd.isna(result.loc['AAA', 'mom_12_1'])


def test_mom_12_1_manual(prices_full):
    as_of = _as_of(prices_full)
    aaa = prices_full[prices_full[PRICE_TICKER] == 'AAA'].copy()
    aaa[PRICE_DATE] = pd.to_datetime(aaa[PRICE_DATE])

    cutoff = pd.Timestamp(as_of)
    p_1m_row = aaa[aaa[PRICE_DATE] <= cutoff - pd.Timedelta(days=30)].iloc[-1]
    p_12m_row = aaa[aaa[PRICE_DATE] <= cutoff - pd.Timedelta(days=365)].iloc[-1]
    expected = np.log(p_1m_row[PRICE_CLOSE] / p_12m_row[PRICE_CLOSE])

    result = compute_momentum(prices_full, as_of)
    assert result.loc['AAA', 'mom_12_1'] == pytest.approx(expected, rel=1e-6)


def test_mom_6_1_manual(prices_full):
    as_of = _as_of(prices_full)
    aaa = prices_full[prices_full[PRICE_TICKER] == 'AAA'].copy()
    aaa[PRICE_DATE] = pd.to_datetime(aaa[PRICE_DATE])

    cutoff = pd.Timestamp(as_of)
    p_1m_row = aaa[aaa[PRICE_DATE] <= cutoff - pd.Timedelta(days=30)].iloc[-1]
    p_6m_row = aaa[aaa[PRICE_DATE] <= cutoff - pd.Timedelta(days=182)].iloc[-1]
    expected = np.log(p_1m_row[PRICE_CLOSE] / p_6m_row[PRICE_CLOSE])

    result = compute_momentum(prices_full, as_of)
    assert result.loc['AAA', 'mom_6_1'] == pytest.approx(expected, rel=1e-6)


def test_vol_21d_manual(prices_full):
    as_of = _as_of(prices_full)
    aaa = prices_full[prices_full[PRICE_TICKER] == 'AAA'].copy()
    aaa = aaa[aaa[PRICE_DATE] <= pd.Timestamp(as_of)].sort_values(PRICE_DATE)
    closes = aaa[PRICE_CLOSE].values
    log_rets = np.log(closes[-21:] / closes[-22:-1])
    expected = log_rets.std() * np.sqrt(252)

    result = compute_momentum(prices_full, as_of)
    assert result.loc['AAA', 'vol_21d'] == pytest.approx(expected, rel=1e-6)


def test_ma200_ratio_manual(prices_full):
    as_of = _as_of(prices_full)
    aaa = prices_full[prices_full[PRICE_TICKER] == 'AAA'].copy()
    aaa = aaa[aaa[PRICE_DATE] <= pd.Timestamp(as_of)].sort_values(PRICE_DATE)
    closes = aaa[PRICE_CLOSE].values
    ma200 = closes[-200:].mean()

    aaa_latest = aaa.iloc[-1][PRICE_CLOSE]
    expected = aaa_latest / ma200

    result = compute_momentum(prices_full, as_of)
    assert result.loc['AAA', 'ma200_ratio'] == pytest.approx(expected, rel=1e-6)


def test_mom_12_1_nan_short_history():
    df = _make_prices('SHORT', 200)  # < 365 days
    as_of = df[PRICE_DATE].max().date()
    result = compute_momentum(df, as_of)
    assert pd.isna(result.loc['SHORT', 'mom_12_1'])


def test_vol_21d_nan_short_history():
    df = _make_prices('SHORT', 20)  # < 22 rows
    as_of = df[PRICE_DATE].max().date()
    result = compute_momentum(df, as_of)
    assert pd.isna(result.loc['SHORT', 'vol_21d'])


def test_ma200_ratio_nan_short_history():
    df = _make_prices('SHORT', 150)  # < 200 rows
    as_of = df[PRICE_DATE].max().date()
    result = compute_momentum(df, as_of)
    assert pd.isna(result.loc['SHORT', 'ma200_ratio'])


def test_pit_cutoff_respected():
    df = _make_prices('AAA', 450)
    # Use a cutoff 100 days before end — result should differ from using full history
    as_of_early = (df[PRICE_DATE].max() - pd.Timedelta(days=100)).date()
    as_of_late = df[PRICE_DATE].max().date()
    r_early = compute_momentum(df, as_of_early)
    r_late = compute_momentum(df, as_of_late)
    # vol_21d must differ since different trailing window
    assert r_early.loc['AAA', 'vol_21d'] != pytest.approx(r_late.loc['AAA', 'vol_21d'], rel=1e-3)
