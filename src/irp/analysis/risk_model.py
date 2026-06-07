"""Pure multi-factor risk-model math for the `/analysis` Factor-model section.

Regress an instrument's period returns on systematic factor returns:

    r(t) = alpha + Σ_k beta_k · F_k(t) + e(t)

No DB, no Dash — plain pandas in/out, unit-tested in `tests/test_risk_model.py`. The
factor-return panel is assembled by `risk_model_service` from the backtest infra.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


def _align(y: pd.Series, X: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Inner-align y and the factor matrix on common dates, dropping NaN rows."""
    joined = pd.concat([y.rename('__y__'), X], axis=1, join='inner').dropna()
    return joined['__y__'], joined[list(X.columns)]


def factor_regression(y: pd.Series, X: pd.DataFrame, ppy: float) -> dict:
    """OLS of returns `y` on factor returns `X`.

    Returns alpha (annualized = per-period intercept × ppy), betas/tvalues per factor,
    r2, residual Series (on the aligned index), and n.
    """
    yv, Xv = _align(y, X)
    cols = list(Xv.columns)
    nan = {'alpha': np.nan, 'betas': {c: np.nan for c in cols},
           'tvalues': {c: np.nan for c in cols}, 'r2': np.nan,
           'resid': pd.Series(dtype=float), 'n': int(len(yv))}
    if len(yv) <= len(cols) + 1:
        return nan
    Xc = sm.add_constant(Xv)
    fit = sm.OLS(yv.to_numpy(), Xc.to_numpy()).fit()
    names = ['const'] + cols
    params = dict(zip(names, fit.params, strict=True))
    tvals = dict(zip(names, fit.tvalues, strict=True))
    resid = pd.Series(fit.resid, index=yv.index)
    return {
        'alpha': float(params['const'] * ppy),
        'betas': {c: float(params[c]) for c in cols},
        'tvalues': {c: float(tvals[c]) for c in cols},
        'r2': float(fit.rsquared),
        'resid': resid,
        'n': int(len(yv)),
    }


def rolling_exposures(y: pd.Series, X: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling multi-factor betas (one column per factor) over a trailing `window`."""
    yv, Xv = _align(y, X)
    cols = list(Xv.columns)
    if len(yv) < window:
        return pd.DataFrame(columns=cols)
    Xc = sm.add_constant(Xv)
    model = RollingOLS(yv.to_numpy(), Xc.to_numpy(), window=window)
    fit = model.fit()
    params = pd.DataFrame(fit.params, index=yv.index, columns=['const'] + cols)
    return params[cols]


def return_contributions(
    y: pd.Series, X: pd.DataFrame, betas: dict, alpha_per_period: float,
) -> pd.Series:
    """Average return decomposition: each factor contributes `beta_k · mean(F_k)` and
    `alpha` is the per-period intercept. With an OLS intercept the mean residual is
    exactly zero, so `alpha + Σ contributions == mean(y)` (no residual term — it would
    always be ~0 and only clutter the chart; the unexplained piece is `alpha`).
    """
    _, Xv = _align(y, X)
    cols = list(Xv.columns)
    parts = {'alpha': float(alpha_per_period)}
    for c in cols:
        parts[c] = float(betas[c] * Xv[c].mean())
    return pd.Series(parts, index=['alpha', *cols])


def risk_contributions(X: pd.DataFrame, betas: dict, resid: pd.Series) -> pd.Series:
    """Variance shares. Systematic per-factor contribution is `beta_k · (Σ·beta)_k`
    (Σ = factor covariance); residual is `var(resid)`. All normalized to total variance
    so the series sums to 1.
    """
    cols = list(X.columns)
    b = np.array([betas[c] for c in cols], dtype=float)
    sigma = X[cols].cov().to_numpy()
    sigma_b = sigma @ b
    sys_contrib = b * sigma_b                      # per-factor variance contribution
    resid_var = float(np.var(resid.to_numpy(), ddof=1)) if len(resid) > 1 else 0.0
    total = float(sys_contrib.sum() + resid_var)
    if total <= 0:
        return pd.Series({**{c: np.nan for c in cols}, 'residual': np.nan},
                         index=[*cols, 'residual'])
    parts = {c: float(sys_contrib[i] / total) for i, c in enumerate(cols)}
    parts['residual'] = resid_var / total
    return pd.Series(parts, index=[*cols, 'residual'])
