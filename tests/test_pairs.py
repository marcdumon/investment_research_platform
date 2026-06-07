import numpy as np
import pandas as pd

from irp.analysis import pairs as pr


def _idx(n):
    return pd.bdate_range('2010-01-01', periods=n)


def test_engle_granger_cointegrated():
    rng = np.random.default_rng(0)
    n = 600
    b = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=_idx(n))   # random walk
    a = 2.0 * b + pd.Series(rng.normal(0, 0.5, n), index=_idx(n))  # cointegrated, hedge 2
    res = pr.engle_granger(a, b)
    assert res['pvalue'] < 0.05
    assert np.isclose(res['hedge_ratio'], 2.0, atol=0.05)
    assert res['n'] == n
    assert len(res['spread']) == n
    assert abs(res['spread'].mean()) < 1e-6        # spread is the OLS residual -> mean 0


def test_engle_granger_not_cointegrated():
    rng = np.random.default_rng(3)
    n = 600
    a = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=_idx(n))
    b = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=_idx(n))   # independent walk
    res = pr.engle_granger(a, b)
    assert res['pvalue'] > 0.05


def test_spread_zscore_standardized():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(5, 3, 500), index=_idx(500))
    z = pr.spread_zscore(s)
    assert np.isclose(z.mean(), 0.0, atol=1e-9)
    assert np.isclose(z.std(ddof=0), 1.0, atol=1e-6)


def test_half_life_known_ar1():
    rng = np.random.default_rng(2)
    n = 5000
    phi = 0.95
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, 1)
    s = pd.Series(x, index=_idx(n))
    hl = pr.half_life(s)
    expected = -np.log(2) / np.log(phi)            # ~13.5
    assert np.isclose(hl, expected, rtol=0.25)
    assert hl > 0


def test_half_life_nan_when_not_mean_reverting():
    rng = np.random.default_rng(7)
    s = pd.Series(np.cumsum(rng.normal(0, 1, 1000)), index=_idx(1000))  # random walk
    hl = pr.half_life(s)
    assert np.isnan(hl) or hl > 200                # no (or absurdly slow) reversion


def test_lead_lag_recovers_shift():
    rng = np.random.default_rng(4)
    n = 2000
    base = rng.normal(0, 1, n)
    rb = pd.Series(base, index=_idx(n))
    k = 3
    ra = pd.Series(np.r_[np.zeros(k), base[:-k]], index=_idx(n))   # a(t) = b(t-k): b leads a
    lags, xcorr, best = pr.lead_lag(ra, rb, max_lag=10)
    assert best == k
    assert len(lags) == len(xcorr)
