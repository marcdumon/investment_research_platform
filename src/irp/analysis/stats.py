"""Pure return-statistics functions for the `/analysis` page.

No DB, no Dash, no global state — every function takes plain pandas/numpy inputs and
returns plain outputs, so they are trivially unit-testable. The UI service slices the
price panel and feeds Series in here.

Conventions:
- Returns are **log returns** unless stated otherwise.
- `ppy` (periods per year) annualizes: 252 daily / 52 weekly / 12 monthly (`PERIODS_PER_YEAR`).
- Alignment is explicit and NaN-dropping is explicit (we report `n` used) — nothing is
  silently imputed.
"""
import numpy as np
import pandas as pd
from scipy import stats as _sp
from statsmodels.tsa.stattools import acf as _acf, adfuller as _adfuller

PERIODS_PER_YEAR = {'D': 252, 'W': 52, 'M': 12}

# pandas resample period-end aliases per frequency code
_RESAMPLE_RULE = {'W': 'W-FRI', 'M': 'ME'}


def resample_close(close: pd.Series, freq: str) -> pd.Series:
    """Resample a close-price Series to period-end (last) values. `'D'` is a no-op."""
    s = close.dropna().sort_index()
    if freq == 'D' or s.empty:
        return s
    rule = _RESAMPLE_RULE.get(freq)
    if rule is None:
        return s
    return s.resample(rule).last().dropna()


def to_log_returns(close: pd.Series) -> pd.Series:
    """Log returns of a close-price Series; first (NaN) row dropped."""
    s = close.dropna().sort_index()
    return np.log(s / s.shift(1)).dropna()


def summary_stats(rets: pd.Series, ppy: int) -> dict:
    """Distribution + risk summary of a return Series."""
    r = rets.dropna()
    n = int(r.size)
    if n == 0:
        return {k: np.nan for k in (
            'ann_return', 'ann_vol', 'sharpe', 'skew', 'excess_kurtosis', 'min', 'max',
            'hit_rate', 'var95', 'cvar95', 'jb_stat', 'jb_p')} | {'n': 0}
    vals = r.to_numpy()
    ann_return = float(vals.mean() * ppy)
    ann_vol = float(vals.std(ddof=1) * np.sqrt(ppy)) if n > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol else np.nan
    var95 = float(np.quantile(vals, 0.05))
    tail = vals[vals <= var95]
    cvar95 = float(tail.mean()) if tail.size else var95
    # A (near-)constant series has no shape — skip the moment calc (scipy warns on it).
    near_constant = vals.std(ddof=0) < 1e-15
    if near_constant:
        skew = excess_kurt = 0.0
        jb_stat = jb_p = np.nan
    else:
        skew = float(_sp.skew(vals))
        excess_kurt = float(_sp.kurtosis(vals))         # Fisher: excess (0 == normal)
        jb_stat, jb_p = (float(x) for x in _sp.jarque_bera(vals)) if n > 2 else (np.nan, np.nan)
    return {
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'skew': skew,
        'excess_kurtosis': excess_kurt,
        'min': float(vals.min()),
        'max': float(vals.max()),
        'hit_rate': float((vals > 0).mean()),
        'var95': var95,
        'cvar95': cvar95,
        'jb_stat': jb_stat,
        'jb_p': jb_p,
        'n': n,
    }


def histogram_normal(rets: pd.Series, bins: int = 60):
    """Server-side histogram (counts + edges) plus a fitted-normal pdf curve scaled to
    the histogram, so the page ships ~60 numbers instead of the whole return series.

    Returns (counts, edges, x_norm, y_norm). y_norm is on the count scale.
    """
    vals = rets.dropna().to_numpy()
    if vals.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    counts, edges = np.histogram(vals, bins=bins)
    mu, sigma = float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    x_norm = np.linspace(edges[0], edges[-1], 200)
    width = edges[1] - edges[0]
    y_norm = (_sp.norm.pdf(x_norm, mu, sigma) * vals.size * width
              if sigma > 0 else np.zeros_like(x_norm))
    return counts, edges, x_norm, y_norm


def qq_points(rets: pd.Series):
    """Normal QQ points via `scipy.stats.probplot`.

    Returns (theoretical_quantiles, ordered_sample, slope, intercept). The reference
    line is `intercept + slope * theoretical`; slope ≈ sample std.
    """
    vals = rets.dropna().to_numpy()
    if vals.size < 2:
        return np.array([]), np.array([]), np.nan, np.nan
    (osm, osr), (slope, intercept, _r) = _sp.probplot(vals, dist='norm')
    return osm, osr, float(slope), float(intercept)


def cumulative(rets: pd.Series) -> pd.Series:
    """Cumulative log return (sum of log returns)."""
    return rets.dropna().cumsum()


def drawdown(rets: pd.Series) -> pd.Series:
    """Underwater curve in log terms: cum log return minus its running peak (≤ 0)."""
    cum = rets.dropna().cumsum()
    return cum - cum.cummax()


def rolling_vol(rets: pd.Series, window: int, ppy: int) -> pd.Series:
    """Annualized rolling standard deviation of returns."""
    return rets.dropna().rolling(window).std(ddof=1) * np.sqrt(ppy)


def autocorr(rets: pd.Series, nlags: int = 40):
    """Autocorrelation function (lags 0..nlags) with 95% confidence interval.

    Returns (acf_vals, confint) — `confint` is an (nlags+1, 2) array of bands centred on
    the acf values (statsmodels convention).
    """
    vals = rets.dropna().to_numpy()
    nlags = min(nlags, max(1, vals.size - 1))
    acf_vals, confint = _acf(vals, nlags=nlags, alpha=0.05, fft=True)
    return acf_vals, confint


def adf(rets: pd.Series):
    """Augmented Dickey-Fuller stationarity test. Returns (stat, pvalue)."""
    vals = rets.dropna().to_numpy()
    if vals.size < 10:
        return np.nan, np.nan
    res = _adfuller(vals, autolag='AIC')
    return float(res[0]), float(res[1])


def market_model(stock_rets: pd.Series, bench_rets: pd.Series, ppy: int) -> dict:
    """OLS market model: stock_ret = alpha + beta * bench_ret + resid.

    Inner-aligns the two Series on common dates. `alpha` is annualized (per-period
    intercept × ppy). Returns beta, alpha, r2, tracking_error (annualized resid std),
    residuals (Series on the aligned index), up/down capture, and n.
    """
    joined = pd.concat([stock_rets, bench_rets], axis=1, join='inner').dropna()
    joined.columns = ['stock', 'bench']
    n = int(len(joined))
    nan = {'beta': np.nan, 'alpha': np.nan, 'r2': np.nan, 'resid_vol': np.nan,
           'up_capture': np.nan, 'down_capture': np.nan, 'residuals': pd.Series(dtype=float),
           'n': n}
    if n < 3:
        return nan
    x = joined['bench'].to_numpy()
    y = joined['stock'].to_numpy()
    reg = _sp.linregress(x, y)
    resid = pd.Series(y - (reg.intercept + reg.slope * x), index=joined.index)
    resid_vol = float(resid.std(ddof=1) * np.sqrt(ppy))   # annualized idiosyncratic vol
    up = joined[joined['bench'] > 0]
    dn = joined[joined['bench'] < 0]
    up_cap = float(up['stock'].mean() / up['bench'].mean()) if len(up) and up['bench'].mean() else np.nan
    dn_cap = float(dn['stock'].mean() / dn['bench'].mean()) if len(dn) and dn['bench'].mean() else np.nan
    return {
        'beta': float(reg.slope),
        'alpha': float(reg.intercept * ppy),
        'r2': float(reg.rvalue ** 2),
        'resid_vol': resid_vol,
        'up_capture': up_cap,
        'down_capture': dn_cap,
        'residuals': resid,
        'slope': float(reg.slope),
        'intercept': float(reg.intercept),
        'bench_aligned': joined['bench'],     # x for the scatter
        'stock_aligned': joined['stock'],     # y for the scatter
        'n': n,
    }


def rolling_beta(stock_rets: pd.Series, bench_rets: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta = rolling cov(stock, bench) / rolling var(bench)."""
    joined = pd.concat([stock_rets, bench_rets], axis=1, join='inner').dropna()
    joined.columns = ['stock', 'bench']
    cov = joined['stock'].rolling(window).cov(joined['bench'])
    var = joined['bench'].rolling(window).var()
    return cov / var
