import numpy as np
import pandas as pd

from irp.analysis import regime as rg


def _series(vals, start='2015-01-01'):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(np.asarray(vals, dtype=float), index=idx)


# ---------------------------------------------------------------- feature panel

def test_build_feature_panel_raw_curve_and_trend():
    n = 300
    y10 = _series(np.full(n, 3.0))
    y2 = _series(np.full(n, 1.0))
    y3m = _series(np.full(n, 0.5))
    equity = _series(np.linspace(100.0, 200.0, n))          # steady uptrend
    feats = rg.build_feature_panel(
        {'y10': y10, 'y2': y2, 'y3m': y3m, 'equity': equity}, standardize=False)
    assert np.allclose(feats['curve_10_2'].dropna(), 2.0)
    assert np.allclose(feats['curve_10_3m'].dropna(), 2.5)
    # uptrend => price above its own MA200 => eq_trend > 0 once the MA exists
    assert feats['eq_trend'].dropna().iloc[-1] > 0
    assert feats['eq_mom'].dropna().iloc[-1] > 0


def test_build_feature_panel_drops_missing_roles():
    equity = _series(np.linspace(100.0, 120.0, 260))
    feats = rg.build_feature_panel({'equity': equity}, standardize=False)
    assert 'eq_trend' in feats.columns
    assert 'curve_10_2' not in feats.columns          # no yield roles supplied
    assert 'usd_trend' not in feats.columns


def test_expanding_z_is_causal():
    # z at time t must use only observations up to t (no look-ahead)
    s = _series(np.arange(1, 401, dtype=float))
    z = rg.expanding_z(s.to_frame('x'), min_periods=10)['x']
    t = 200
    hist = s.iloc[: t + 1]
    expect = (s.iloc[t] - hist.mean()) / hist.std()
    assert np.isclose(z.iloc[t], expect)


# ---------------------------------------------------------------- rule classifier

def test_rule_regime_flags_risk_off():
    idx = pd.bdate_range('2015-01-01', periods=5)
    z = pd.DataFrame(
        {'eq_trend': -3.0, 'eq_mom': -3.0, 'curve_10_2': -3.0, 'vol': 3.0},
        index=idx)
    out = rg.rule_regime(z)
    assert (out['label'] == 'risk_off').all()
    assert (out['risk_score'] < 40).all()


def test_rule_regime_flags_risk_on():
    idx = pd.bdate_range('2015-01-01', periods=5)
    z = pd.DataFrame(
        {'eq_trend': 3.0, 'eq_mom': 3.0, 'curve_10_2': 3.0, 'vol': -3.0},
        index=idx)
    out = rg.rule_regime(z)
    assert (out['label'] == 'risk_on').all()
    assert (out['risk_score'] > 60).all()


def test_rule_regime_score_bounded():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2015-01-01', periods=50)
    z = pd.DataFrame(rng.normal(0, 5, (50, 4)),
                     columns=['eq_trend', 'eq_mom', 'vol', 'usd_trend'], index=idx)
    out = rg.rule_regime(z)
    assert out['risk_score'].between(0, 100).all()


def test_feature_contributions_signs_by_weight():
    idx = pd.bdate_range('2015-01-01', periods=3)
    z = pd.DataFrame({'eq_trend': 2.0, 'vol': 2.0}, index=idx)   # vol weight is negative
    c = rg.feature_contributions(z)
    assert c['eq_trend'] > 0                                     # +weight, +z -> risk-on
    assert c['vol'] < 0                                          # -weight, +z -> risk-off
    assert list(c.index) == list(c.sort_values().index)         # ascending


# ---------------------------------------------------------------- HMM

def test_hmm_regime_recovers_two_states():
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.bdate_range('2010-01-01', periods=2 * n)
    calm = rng.normal(0.0, 0.003, n)
    crisis = rng.normal(0.0, 0.03, n)
    feats = pd.DataFrame(
        {'eq_mom': np.concatenate([calm, crisis]),
         'vol': np.concatenate([np.full(n, 0.003), np.full(n, 0.03)])},
        index=idx)
    fit = rg.hmm_regime(feats, n_states=2, mode='full', seed=0)
    lab = fit.labels
    assert lab.nunique() == 2
    first = lab.iloc[:n].mode().iloc[0]
    second = lab.iloc[n:].mode().iloc[0]
    assert first != second                              # two halves = two regimes
    assert fit.transition.shape == (2, 2)
    assert np.allclose(fit.transition.sum(axis=1), 1.0)


def test_transition_matrix_rows_sum_to_one():
    lab = pd.Series([0, 0, 1, 1, 0, 1, 1, 1, 0, 0])
    tm = rg.transition_matrix(lab)
    assert np.allclose(tm.sum(axis=1), 1.0)
    assert tm.loc[1, 1] > 0                              # 1->1 transitions exist


# ---------------------------------------------------------------- conditioning glue

def test_align_to_dates_is_backward_asof():
    daily = pd.Series(['risk_off', 'risk_on'],
                      index=pd.to_datetime(['2015-01-05', '2015-03-01']))
    dates = pd.to_datetime(['2015-02-01', '2015-04-01'])
    out = rg.align_to_dates(daily, dates)
    assert list(out) == ['risk_off', 'risk_on']         # most-recent prior label


def test_regime_conditioned_ic_groups():
    idx = pd.to_datetime(['2015-01-01', '2015-02-01', '2015-03-01', '2015-04-01'])
    ic = pd.Series([0.10, 0.20, -0.05, -0.15], index=idx)
    labels = pd.Series(['risk_on', 'risk_on', 'risk_off', 'risk_off'], index=idx)
    out = rg.regime_conditioned_ic(ic, labels)
    assert np.isclose(out.loc['risk_on', 'mean_ic'], 0.15)
    assert np.isclose(out.loc['risk_off', 'mean_ic'], -0.10)
    assert out.loc['risk_on', 'n'] == 2


def test_gated_returns_zeroes_disallowed():
    idx = pd.bdate_range('2015-01-01', periods=4)
    rets = pd.Series([0.01, 0.02, -0.03, 0.04], index=idx)
    labels = pd.Series(['risk_on', 'risk_off', 'risk_off', 'risk_on'], index=idx)
    out = rg.gated_returns(rets, labels, allowed={'risk_on'})
    assert np.allclose(out['gated_period'].to_numpy(), [0.01, 0.0, 0.0, 0.04])
    assert np.allclose(out['base_period'].to_numpy(), rets.to_numpy())


def _trend(rate, n=80, start=100.0):
    idx = pd.bdate_range('2015-01-01', periods=n)
    return pd.Series(start * (1.0 + rate) ** np.arange(n), index=idx)


def test_return_states_buckets_by_threshold():
    bull = rg.return_states(_trend(0.01), lookback=20, threshold=0.05)
    bear = rg.return_states(_trend(-0.01), lookback=20, threshold=0.05)
    flat = rg.return_states(_trend(0.0), lookback=20, threshold=0.05)
    assert (bull == 'bull').all()
    assert (bear == 'bear').all()
    assert (flat == 'sideways').all()


def test_return_states_nonoverlap_is_sparser():
    s = _trend(0.005, n=200)
    over = rg.return_states(s, lookback=20, overlapping=True)
    non = rg.return_states(s, lookback=20, overlapping=False)
    assert len(non) < len(over)
    assert len(non) <= len(over) // 10 + 2          # ~ every 20th obs


def test_n_step_distribution_compounds():
    P = pd.DataFrame([[0.8, 0.2], [0.5, 0.5]], index=['bear', 'bull'], columns=['bear', 'bull'])
    dist = rg.n_step_distribution(P, 'bull', steps=2)
    assert np.allclose(dist.loc[1].to_numpy(), [0.5, 0.5])           # one step = P row
    assert np.allclose(dist.loc[2].to_numpy(), [0.65, 0.35])         # [.5,.5] @ P
    assert np.allclose(dist.sum(axis=1).to_numpy(), 1.0)


def test_stationary_distribution_matches_power_limit():
    P = pd.DataFrame([[0.9, 0.1], [0.4, 0.6]], index=['bear', 'bull'], columns=['bear', 'bull'])
    pi = rg.stationary_distribution(P)
    assert np.isclose(pi.sum(), 1.0)
    # πP = π
    assert np.allclose((pi.to_numpy() @ P.to_numpy()), pi.to_numpy(), atol=1e-6)


def test_directional_signal_sign():
    P = pd.DataFrame([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]],
                     index=['bear', 'sideways', 'bull'], columns=['bear', 'sideways', 'bull'])
    assert np.isclose(rg.directional_signal(P, 'bull'), 0.7 - 0.1)   # bull row: P(bull)-P(bear)
    assert rg.directional_signal(P, 'bear') < 0                      # bear row: bear-heavy


def test_markov_backtest_keys_and_long_bias():
    s = _trend(0.01, n=200)                       # steady uptrend -> long bias profits
    res = rg.markov_backtest(s, lookback=20, threshold=0.05, horizon=5)
    for k in ('strat_cumret', 'hold_cumret', 'strat_sharpe', 'hold_sharpe'):
        assert k in res
    assert not res['strat_cumret'].empty


def test_tactical_table_ranks_leader_first():
    n = 300
    idx = pd.bdate_range('2015-01-01', periods=n)
    leader = pd.Series(np.linspace(100, 300, n), index=idx)      # strongest trend
    flat = pd.Series(np.full(n, 100.0), index=idx)
    laggard = pd.Series(np.linspace(100, 90, n), index=idx)
    tab = rg.tactical_table({'LEAD': leader, 'FLAT': flat, 'LAG': laggard},
                            lookbacks=(63, 126))
    assert tab.loc['LEAD', 'rank'] == 1
    assert tab.loc['LAG', 'rank'] == tab['rank'].max()
