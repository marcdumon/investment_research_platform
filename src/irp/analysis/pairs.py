"""Pure pairs / cointegration math for the `/analysis` Pair section.

Statistical-arbitrage diagnostics on two instruments A & B: cointegration of their
(log-)price spread, hedge ratio, spread z-score, mean-reversion half-life, and lead-lag
cross-correlation. No DB, no Dash — plain pandas in/out, unit-tested in
`tests/test_pairs.py`.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    j = pd.concat([a.rename('a'), b.rename('b')], axis=1, join='inner').dropna()
    return j['a'], j['b']


def engle_granger(a: pd.Series, b: pd.Series) -> dict:
    """Engle-Granger cointegration of A on B.

    Hedge ratio from OLS `a = const + beta·b`; the spread is the regression residual
    `a − const − beta·b`. Cointegration p-value via `statsmodels.tsa.stattools.coint`.
    """
    av, bv = _align(a, b)
    n = int(len(av))
    nan = {'coint_t': np.nan, 'pvalue': np.nan, 'hedge_ratio': np.nan, 'const': np.nan,
           'spread': pd.Series(dtype=float), 'n': n}
    if n < 20:
        return nan
    X = sm.add_constant(bv.to_numpy())
    fit = sm.OLS(av.to_numpy(), X).fit()
    const, beta = float(fit.params[0]), float(fit.params[1])
    spread = pd.Series(av.to_numpy() - const - beta * bv.to_numpy(), index=av.index)
    coint_t, pvalue, _crit = coint(av.to_numpy(), bv.to_numpy())
    return {'coint_t': float(coint_t), 'pvalue': float(pvalue), 'hedge_ratio': beta,
            'const': const, 'spread': spread, 'n': n}


def spread_zscore(spread: pd.Series, window: int | None = None) -> pd.Series:
    """Standardized spread. Full-sample when `window` is None, else rolling."""
    s = spread.dropna()
    if window:
        mu = s.rolling(window).mean()
        sd = s.rolling(window).std(ddof=0)
        return (s - mu) / sd
    sd = s.std(ddof=0)
    if sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life (Ornstein-Uhlenbeck): regress `Δspread` on `lag(spread)`
    (`Δs = c + λ·s_{t-1}`); `half_life = −ln 2 / λ`. NaN when `λ ≥ 0` (not reverting)."""
    s = spread.dropna()
    if len(s) < 10:
        return np.nan
    lag = s.shift(1)
    delta = s - lag
    df = pd.concat([delta.rename('d'), lag.rename('l')], axis=1).dropna()
    X = sm.add_constant(df['l'].to_numpy())
    lam = float(sm.OLS(df['d'].to_numpy(), X).fit().params[1])
    if lam >= 0:
        return np.nan
    return float(-np.log(2) / lam)


def lead_lag(ra: pd.Series, rb: pd.Series, max_lag: int = 10):
    """Cross-correlation of return series A vs B over ±`max_lag`.

    `xcorr[lag] = corr(ra(t), rb(t−lag))`, so a positive `best_lag` means B leads A.
    Returns (lags, xcorr, best_lag) where `best_lag = argmax|xcorr|`.
    """
    rav, rbv = _align(ra, rb)
    lags = list(range(-max_lag, max_lag + 1))
    xs = []
    for lag in lags:
        c = rav.corr(rbv.shift(lag))
        xs.append(float(c) if c == c else 0.0)
    xcorr = np.array(xs)
    best = lags[int(np.argmax(np.abs(xcorr)))]
    return lags, xcorr, best
