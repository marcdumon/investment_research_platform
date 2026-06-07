import numpy as np
import pandas as pd

from irp.analysis import risk_model as rm


def _factors(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2010-01-01', periods=n)
    X = pd.DataFrame({
        'market': rng.normal(0, 0.04, n),
        'value': rng.normal(0, 0.03, n),
        'quality': rng.normal(0, 0.02, n),
        'momentum': rng.normal(0, 0.03, n),
    }, index=idx)
    return X, rng


def test_factor_regression_recovers_betas():
    X, rng = _factors()
    y = 1.5 * X['market'] + 0.5 * X['value'] + rng.normal(0, 0.002, len(X))
    res = rm.factor_regression(y, X, ppy=4)
    assert np.isclose(res['betas']['market'], 1.5, atol=0.05)
    assert np.isclose(res['betas']['value'], 0.5, atol=0.05)
    assert abs(res['betas']['quality']) < 0.1
    assert res['r2'] > 0.95
    assert res['n'] == len(X)
    assert abs(res['alpha']) < 0.05                       # annualized, ~0
    assert len(res['resid']) == len(X)
    assert res['tvalues']['market'] > res['tvalues']['quality']   # real loading more significant


def test_factor_regression_inner_aligns():
    X, rng = _factors(n=300)
    y = (X['market'] + rng.normal(0, 0.001, len(X))).iloc[:200]   # shorter
    res = rm.factor_regression(y, X, ppy=4)
    assert res['n'] == 200


def test_return_contributions_sum_to_mean():
    X, rng = _factors()
    y = 1.5 * X['market'] + 0.5 * X['value'] + rng.normal(0, 0.002, len(X))
    res = rm.factor_regression(y, X, ppy=4)
    contrib = rm.return_contributions(y, X, res['betas'], res['alpha'] / 4)
    assert set(contrib.index) == {'alpha', 'market', 'value', 'quality', 'momentum'}
    # OLS intercept => mean residual is exactly 0, so contributions sum to mean(y)
    assert np.isclose(contrib.sum(), float(y.mean()), atol=1e-9)
    assert np.isclose(contrib['market'], 1.5 * float(X['market'].mean()), atol=1e-9)


def test_risk_contributions_sum_to_one():
    X, rng = _factors()
    y = 1.5 * X['market'] + 0.5 * X['value'] + rng.normal(0, 0.002, len(X))
    res = rm.factor_regression(y, X, ppy=4)
    risk = rm.risk_contributions(X, res['betas'], res['resid'])
    assert set(risk.index) == {'market', 'value', 'quality', 'momentum', 'residual'}
    assert np.isclose(risk.sum(), 1.0, atol=1e-6)
    assert risk['market'] > risk['value'] > risk['quality']      # market dominates variance
    assert risk['residual'] < 0.1                                # noise small


def test_rolling_exposures_shape_and_level():
    X, rng = _factors(n=300)
    y = 1.5 * X['market'] + 0.5 * X['value'] + rng.normal(0, 0.002, len(X))
    win = 120
    roll = rm.rolling_exposures(y, X, window=win)
    assert list(roll.columns) == ['market', 'value', 'quality', 'momentum']
    assert roll.dropna().shape[0] == len(X) - win + 1
    assert np.isclose(roll['market'].dropna().mean(), 1.5, atol=0.1)
    assert np.isclose(roll['value'].dropna().mean(), 0.5, atol=0.1)
