import numpy as np
import pandas as pd

from irp.analysis import stats as st


def _dates(n):
    return pd.bdate_range('2015-01-01', periods=n)


def test_to_log_returns_basic():
    close = pd.Series([100.0, 110.0, 121.0], index=_dates(3))
    r = st.to_log_returns(close)
    assert len(r) == 2                              # first row dropped
    assert np.isclose(r.iloc[0], np.log(110 / 100))
    assert np.isclose(r.iloc[1], np.log(121 / 110))


def test_resample_close_shrinks_with_coarser_freq():
    close = pd.Series(np.arange(1, 261, dtype=float), index=_dates(260))
    d = st.resample_close(close, 'D')
    w = st.resample_close(close, 'W')
    m = st.resample_close(close, 'M')
    assert len(d) == 260                            # daily is a no-op
    assert len(w) < len(d)
    assert len(m) < len(w)
    assert m.iloc[-1] == close.iloc[-1]             # period-end = last value


def test_summary_stats_normal_vs_lognormal():
    rng = np.random.default_rng(0)
    normal = pd.Series(rng.normal(0, 0.01, 4000), index=_dates(4000))
    s = st.summary_stats(normal, ppy=252)
    assert abs(s['skew']) < 0.2
    assert abs(s['excess_kurtosis']) < 0.3
    assert s['n'] == 4000
    assert s['var95'] < 0                           # a loss quantile
    # right-skewed input -> positive skew
    skewed = pd.Series(rng.lognormal(0, 0.5, 4000) - 1.5, index=_dates(4000))
    assert st.summary_stats(skewed, ppy=252)['skew'] > 0.5


def test_summary_stats_annualization():
    rets = pd.Series(np.full(252, 0.001), index=_dates(252))
    s = st.summary_stats(rets, ppy=252)
    assert np.isclose(s['ann_return'], 0.001 * 252)
    assert np.isclose(s['ann_vol'], 0.0, atol=1e-12)  # constant -> ~zero vol
    assert s['hit_rate'] == 1.0


def test_market_model_recovers_beta_alpha():
    rng = np.random.default_rng(1)
    idx = _dates(2000)
    bench = pd.Series(rng.normal(0, 0.01, 2000), index=idx)
    stock = 2.0 * bench + rng.normal(0, 0.0005, 2000)   # beta 2, alpha ~0
    mm = st.market_model(stock, bench, ppy=252)
    assert np.isclose(mm['beta'], 2.0, atol=0.05)
    assert abs(mm['alpha']) < 0.05                       # annualized, ~0
    assert mm['r2'] > 0.9
    assert mm['n'] == 2000
    assert len(mm['residuals']) == 2000


def test_market_model_aligns_on_common_dates():
    a = pd.Series([0.01, 0.02, -0.01, 0.03], index=_dates(4))
    b = pd.Series([0.01, 0.02, -0.01], index=_dates(3))   # shorter
    mm = st.market_model(a, b, ppy=252)
    assert mm['n'] == 3                                   # inner-aligned


def test_rolling_beta_tracks_two():
    rng = np.random.default_rng(2)
    idx = _dates(500)
    bench = pd.Series(rng.normal(0, 0.01, 500), index=idx)
    stock = 2.0 * bench + rng.normal(0, 0.0003, 500)
    rb = st.rolling_beta(stock, bench, window=60)
    assert np.isclose(rb.dropna().mean(), 2.0, atol=0.1)


def test_drawdown_and_cumulative_known_path():
    # prices 100 -> 120 -> 60 -> 90 ; log returns between them
    close = pd.Series([100.0, 120.0, 60.0, 90.0], index=_dates(4))
    rets = st.to_log_returns(close)
    dd = st.drawdown(rets)
    assert np.isclose(dd.min(), np.log(60 / 120))        # worst underwater = peak 120 -> 60
    cum = st.cumulative(rets)
    assert np.isclose(cum.iloc[-1], np.log(90 / 100))    # total log return


def test_qq_points_slope_is_sample_std():
    rng = np.random.default_rng(3)
    rets = pd.Series(rng.normal(0, 0.02, 3000), index=_dates(3000))
    theo, samp, slope, intercept = st.qq_points(rets)
    assert len(theo) == len(samp) == 3000
    assert np.isclose(slope, rets.std(ddof=1), rtol=0.1)  # probplot slope ~ std


def test_autocorr_white_noise_within_bands():
    rng = np.random.default_rng(4)
    rets = pd.Series(rng.normal(0, 1, 3000), index=_dates(3000))
    acf_vals, confint = st.autocorr(rets, nlags=20)
    assert len(acf_vals) == 21                            # lag 0..20
    assert acf_vals[0] == 1.0
    # most lags 1.. within the confidence band
    lo = confint[1:, 0] - acf_vals[1:]
    hi = confint[1:, 1] - acf_vals[1:]
    inside = (acf_vals[1:] >= lo) & (acf_vals[1:] <= hi)
    assert inside.mean() > 0.8


def test_rolling_vol_annualized():
    rng = np.random.default_rng(5)
    rets = pd.Series(rng.normal(0, 0.01, 300), index=_dates(300))
    rv = st.rolling_vol(rets, window=63, ppy=252)
    expected = rets.iloc[-63:].std(ddof=1) * np.sqrt(252)
    assert np.isclose(rv.iloc[-1], expected)


def test_histogram_normal_shapes():
    rng = np.random.default_rng(6)
    rets = pd.Series(rng.normal(0, 0.01, 5000), index=_dates(5000))
    counts, edges, x_norm, y_norm = st.histogram_normal(rets, bins=60)
    assert len(counts) == 60
    assert len(edges) == 61
    assert len(x_norm) == len(y_norm)
    assert counts.sum() == 5000


def test_adf_stationary_returns():
    rng = np.random.default_rng(7)
    rets = pd.Series(rng.normal(0, 0.01, 1000), index=_dates(1000))
    stat, pval = st.adf(rets)
    assert pval < 0.05                                   # returns are stationary
