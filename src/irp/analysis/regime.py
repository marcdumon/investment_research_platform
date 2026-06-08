"""Pure macro-regime math for the `/regime` page.

Cross-asset state from yields/equity/USD/commodities/vol/crypto closes, classified two
ways: a transparent rule-based risk score and (in `hmm_regime`) a statistical HMM. Plus
glue to feed the regime back into stock decisions (conditioned IC, gated returns) and a
cross-asset tactical table. No DB, no Dash, no look-ahead: every standardization is
expanding-window. Unit-tested in `tests/test_regime.py`.
"""
import dataclasses

import numpy as np
import pandas as pd

# Default signed weights for the rule score: positive = pushes toward risk-on.
# Hand-set interpretable priors (NOT fitted) — overridable via the `weights` arg.
DEFAULT_WEIGHTS: dict[str, float] = {
    'curve_10_2':    1.0,    # steep curve = expansion / risk-on
    'curve_10_3m':   0.5,
    'eq_trend':      1.5,    # price above MA200
    'eq_mom':        1.0,
    'vol':          -1.5,    # high realized vol = risk-off
    'usd_trend':    -0.5,    # strong USD = risk-off
    'commod_trend':  0.5,    # reflation
    'risk_appetite': 0.5,    # crypto bid = risk-on
}
RISK_OFF_BELOW = 40.0
RISK_ON_ABOVE = 60.0


def _aligned(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """Union-index the role closes onto one business-day frame, forward-filled.

    Macro series trade on different calendars (yields vs equity vs FX); ffill carries
    the last observation so per-row differences (curve slope) are well defined.
    """
    wide = pd.concat({k: v.dropna() for k, v in closes.items()}, axis=1).sort_index()
    return wide.ffill()


def build_feature_panel(closes: dict[str, pd.Series], *, standardize: bool = True,
                        zwin_min: int = 252) -> pd.DataFrame:
    """Daily macro feature panel from a `role -> close Series` map.

    Roles consumed when present: `y10/y2/y3m` (Treasury yields, levels), `equity`
    (index level), `usd`, `commod`, `crypto`. Features whose roles are absent are
    silently skipped. With `standardize=True` every feature is expanding-z-scored
    (`zwin_min` burn-in) so the panel is point-in-time; pass `False` for raw values.
    """
    w = _aligned(closes)
    feats: dict[str, pd.Series] = {}
    if 'y10' in w and 'y2' in w:
        feats['curve_10_2'] = w['y10'] - w['y2']
    if 'y10' in w and 'y3m' in w:
        feats['curve_10_3m'] = w['y10'] - w['y3m']
    if 'equity' in w:
        eq = w['equity']
        feats['eq_trend'] = eq / eq.rolling(200).mean() - 1.0
        feats['eq_mom'] = np.log(eq / eq.shift(126))
        feats['vol'] = np.log(eq / eq.shift(1)).rolling(21).std()
    if 'usd' in w:
        feats['usd_trend'] = np.log(w['usd'] / w['usd'].shift(63))
    if 'commod' in w:
        feats['commod_trend'] = np.log(w['commod'] / w['commod'].shift(63))
    if 'crypto' in w:
        feats['risk_appetite'] = np.log(w['crypto'] / w['crypto'].shift(63))
    df = pd.DataFrame(feats).dropna(how='all')
    if standardize:
        df = expanding_z(df, min_periods=zwin_min)
    return df


def expanding_z(df: pd.DataFrame, min_periods: int = 252) -> pd.DataFrame:
    """Causal (expanding-window) z-score per column; rows before `min_periods` are NaN."""
    mean = df.expanding(min_periods=min_periods).mean()
    std = df.expanding(min_periods=min_periods).std()
    return (df - mean) / std.replace(0.0, np.nan)


def rule_regime(features: pd.DataFrame, weights: dict[str, float] | None = None,
                clip: float = 3.0) -> pd.DataFrame:
    """Transparent risk score (0–100) + discrete label from standardized features.

    Weighted average of clipped feature z-scores (only the features present), mapped
    through a logistic to [0, 100]; `< 40` → `risk_off`, `> 60` → `risk_on`, else
    `neutral`. Rows with no usable feature get NaN score / `unknown`.
    """
    w = weights or DEFAULT_WEIGHTS
    cols = [c for c in features.columns if c in w]
    z = features[cols].clip(-clip, clip)
    wv = pd.Series({c: w[c] for c in cols})
    wsum = wv.abs().sum()
    raw = (z * wv).sum(axis=1, min_count=1) / wsum if wsum else pd.Series(np.nan, index=z.index)
    score = 100.0 / (1.0 + np.exp(-raw))
    label = pd.Series('neutral', index=score.index, dtype=object)
    label[score < RISK_OFF_BELOW] = 'risk_off'
    label[score > RISK_ON_ABOVE] = 'risk_on'
    label[score.isna()] = 'unknown'
    return pd.DataFrame({'risk_score': score, 'label': label})


def feature_contributions(features: pd.DataFrame, weights: dict[str, float] | None = None,
                          clip: float = 3.0) -> pd.Series:
    """Signed risk-on contribution (`weight × clipped z`) of the latest row, ascending.

    The per-feature decomposition of `rule_regime`'s score for the most recent date —
    positive pushes toward risk-on, negative toward risk-off."""
    w = weights or DEFAULT_WEIGHTS
    valid = features.dropna()
    if valid.empty:
        return pd.Series(dtype=float)
    row = valid.iloc[-1]
    contrib = {c: float(np.clip(row[c], -clip, clip)) * w[c]
               for c in row.index if c in w}
    return pd.Series(contrib).sort_values()


@dataclasses.dataclass(frozen=True)
class RegimeFit:
    """Result of an HMM fit: integer state labels (0 = most risk-off), the state
    transition matrix, per-state feature means, and smoothed state probabilities."""
    labels: pd.Series
    transition: pd.DataFrame
    state_means: pd.DataFrame
    smoothed_prob: pd.DataFrame
    n_states: int


def _order_key(cols) -> str:
    """Feature used to order HMM states risk-off -> risk-on (low -> high)."""
    for c in ('eq_mom', 'eq_trend'):
        if c in cols:
            return c
    return list(cols)[0]


def _fit_hmm(X: np.ndarray, n_states: int, seed: int):
    from hmmlearn.hmm import GaussianHMM
    m = GaussianHMM(n_components=n_states, covariance_type='diag',
                    n_iter=100, random_state=seed)
    m.fit(X)
    return m


def hmm_regime(features: pd.DataFrame, n_states: int = 3, mode: str = 'full',
               seed: int = 0, refit_every: int = 21, min_train: int = 504) -> RegimeFit:
    """Gaussian-HMM regime states from the standardized feature matrix.

    `mode='full'` fits once on the whole sample (in-sample; use for the display
    timeline). `mode='expanding'` refits every `refit_every` rows on data up to each
    point and keeps the point-in-time state (no look-ahead; use for conditioning).
    States are relabeled so 0 is the most risk-off (lowest mean `eq_mom`/`eq_trend`).
    """
    feats = features.dropna()
    cols = list(feats.columns)
    if feats.empty or len(feats) < n_states:
        empty_p = pd.DataFrame(index=feats.index)
        return RegimeFit(pd.Series(dtype=int), pd.DataFrame(), pd.DataFrame(),
                         empty_p, n_states)
    X = feats.to_numpy()
    key = _order_key(cols)
    key_i = cols.index(key)

    if mode == 'expanding':
        raw = np.full(len(feats), -1, dtype=int)
        model = None
        for i in range(len(feats)):
            if i >= min_train and (model is None or i % refit_every == 0):
                model = _fit_hmm(X[: i + 1], n_states, seed)
            if model is not None:
                raw[i] = int(model.predict(X[: i + 1])[-1])
        model = model or _fit_hmm(X, n_states, seed)
        post = model.predict_proba(X)
    else:
        model = _fit_hmm(X, n_states, seed)
        raw = model.predict(X)
        post = model.predict_proba(X)

    # relabel states 0..k-1 by ascending mean of the ordering feature
    order = np.argsort([X[raw == s, key_i].mean() if (raw == s).any() else np.inf
                        for s in range(n_states)])
    remap = {int(old): new for new, old in enumerate(order)}
    labels = pd.Series([remap.get(int(s), -1) for s in raw], index=feats.index)
    means = (feats.assign(_s=labels).groupby('_s')[cols].mean()
             .reindex(range(n_states)))
    means.index.name = 'state'
    prob = pd.DataFrame(post[:, order], columns=range(n_states), index=feats.index)
    return RegimeFit(labels, transition_matrix(labels, n_states), means, prob, n_states)


def transition_matrix(labels: pd.Series, n_states: int | None = None) -> pd.DataFrame:
    """Empirical state-transition matrix (row i = P(next state | current=i))."""
    lab = labels.dropna()
    if n_states is not None:                              # HMM int states (may carry −1 burn-in)
        lab = lab.astype(int)
        states = list(range(n_states))
    else:                                                 # generic labels (e.g. bull/sideways/bear)
        states = sorted(lab.unique())
    sset = set(states)
    tm = pd.DataFrame(0.0, index=states, columns=states)
    for a, b in zip(lab.iloc[:-1], lab.iloc[1:], strict=True):
        if a in sset and b in sset:                       # skip burn-in (-1) states
            tm.loc[a, b] += 1.0
    row = tm.sum(axis=1)
    return tm.div(row.where(row > 0, 1.0), axis=0)


def align_to_dates(daily_labels: pd.Series, dates) -> pd.Series:
    """Backward as-of map of a daily label series onto `dates` (most-recent prior)."""
    s = daily_labels.dropna().sort_index()
    target = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    return s.reindex(s.index.union(target)).ffill().reindex(target)


def regime_conditioned_ic(ic_series: pd.Series, labels_at_dates: pd.Series) -> pd.DataFrame:
    """Group an IC series by regime label -> `mean_ic`, `icir`, `n` per regime."""
    j = pd.concat([ic_series.rename('ic'), labels_at_dates.rename('regime')],
                  axis=1).dropna()
    g = j.groupby('regime')['ic']
    out = pd.DataFrame({'mean_ic': g.mean(), 'icir': g.mean() / g.std(), 'n': g.size()})
    return out


def gated_returns(period_rets: pd.Series, labels_at_dates: pd.Series,
                  allowed: set, ppy: int = 252) -> dict:
    """Zero out periods whose regime is not in `allowed`; compare to always-on.

    Returns per-period series + cumulative (log) + annualized Sharpe + max drawdown
    for both the gated and the base (always-on) track.
    """
    j = pd.concat([period_rets.rename('r'), labels_at_dates.rename('regime')],
                  axis=1).dropna()
    mask = j['regime'].isin(allowed)
    gated = j['r'].where(mask, 0.0)

    def _sharpe(r: pd.Series) -> float:
        sd = r.std()
        return float(r.mean() / sd * np.sqrt(ppy)) if sd and sd > 0 else np.nan

    def _maxdd(r: pd.Series) -> float:
        cum = r.cumsum()
        return float((cum - cum.cummax()).min())

    return {
        'gated_period': gated, 'base_period': j['r'],
        'gated_cumret': gated.cumsum(), 'base_cumret': j['r'].cumsum(),
        'gated_sharpe': _sharpe(gated), 'base_sharpe': _sharpe(j['r']),
        'gated_maxdd': _maxdd(gated), 'base_maxdd': _maxdd(j['r']),
    }


# ── single-asset Markov regime (the "hedge-fund method") ──────────────
_STATE_ORDER = ['bear', 'sideways', 'bull']


def return_states(close: pd.Series, lookback: int = 20, threshold: float = 0.05,
                  overlapping: bool = True) -> pd.Series:
    """Discrete bull/sideways/bear states from trailing `lookback`-day return.

    `bull` when the trailing return ≥ `threshold`, `bear` when ≤ −`threshold`, else
    `sideways`. With `overlapping=False` the series is sampled every `lookback` days so
    adjacent windows don't share data (avoids the inflated-persistence artifact of the
    daily-overlapping version)."""
    c = close.dropna()
    roll = (c / c.shift(lookback) - 1.0).dropna()
    if not overlapping:
        roll = roll.iloc[::lookback]
    lab = pd.Series('sideways', index=roll.index, dtype=object)
    lab[roll >= threshold] = 'bull'
    lab[roll <= -threshold] = 'bear'
    return lab


def n_step_distribution(P: pd.DataFrame, current: str, steps: int = 10) -> pd.DataFrame:
    """State-probability distribution 1…`steps` days ahead, from `current` state.

    Repeated vector·matrix products (squaring the transition matrix); row `k` is the
    distribution `k` steps out and sums to 1."""
    states = list(P.columns)
    vec = np.array([1.0 if s == current else 0.0 for s in states])
    Pm = P.to_numpy()
    rows = {}
    for k in range(1, steps + 1):
        vec = vec @ Pm
        rows[k] = vec.copy()
    return pd.DataFrame(rows, index=states).T


def stationary_distribution(P: pd.DataFrame) -> pd.Series:
    """Long-run state distribution (πP = π) via power iteration."""
    states = list(P.columns)
    Pm = P.to_numpy()
    vec = np.full(len(states), 1.0 / len(states))
    for _ in range(1000):
        nxt = vec @ Pm
        if np.allclose(nxt, vec, atol=1e-12):
            break
        vec = nxt
    return pd.Series(vec / vec.sum(), index=states)


def directional_signal(P: pd.DataFrame, current: str, up: str = 'bull',
                       down: str = 'bear') -> float:
    """`P(up next) − P(down next)` for the `current` state — the trade signal. Positive
    ⇒ long bias, magnitude ⇒ conviction."""
    if current not in P.index:
        return np.nan
    row = P.loc[current]
    return float(row.get(up, 0.0) - row.get(down, 0.0))


def markov_backtest(close: pd.Series, lookback: int = 20, threshold: float = 0.05,
                    overlapping: bool = True, horizon: int = 5,
                    ppy: int = 252) -> dict:
    """Walk-forward backtest of the `directional_signal` strategy vs buy-and-hold.

    At each state observation the transition matrix is rebuilt from **only prior** states
    (no look-ahead), the signal is read for the current state, and a position in
    [−1, 1] (the clipped signal) earns the next `horizon`-day forward return. Returns
    per-step + cumulative log returns and annualized Sharpe for strategy and hold."""
    c = close.dropna()
    states = return_states(c, lookback, threshold, overlapping)
    if len(states) < 10:
        empty = pd.Series(dtype=float)
        return {'strat_period': empty, 'hold_period': empty, 'strat_cumret': empty,
                'hold_cumret': empty, 'strat_sharpe': np.nan, 'hold_sharpe': np.nan}
    fwd = np.log(c.shift(-horizon) / c)
    s_rows, h_rows, idx = [], [], []
    seq = states.tolist()
    for i in range(2, len(states) - 1):
        P = transition_matrix(states.iloc[: i + 1])
        if P.empty:
            continue
        sig = directional_signal(P, seq[i])
        if not np.isfinite(sig):
            continue
        t = states.index[i]
        fr = fwd.get(t, np.nan)
        if not np.isfinite(fr):
            continue
        pos = float(np.clip(sig, -1.0, 1.0))
        s_rows.append(pos * fr)
        h_rows.append(fr)
        idx.append(t)
    sp = pd.Series(s_rows, index=idx)
    hp = pd.Series(h_rows, index=idx)
    scale = ppy / horizon

    def _sharpe(r):
        sd = r.std()
        return float(r.mean() / sd * np.sqrt(scale)) if sd and sd > 0 else np.nan

    return {'strat_period': sp, 'hold_period': hp,
            'strat_cumret': sp.cumsum(), 'hold_cumret': hp.cumsum(),
            'strat_sharpe': _sharpe(sp), 'hold_sharpe': _sharpe(hp)}


def tactical_table(closes: dict[str, pd.Series], lookbacks=(63, 126, 252)) -> pd.DataFrame:
    """Cross-asset trend table: per asset the log return over each lookback, an average
    score, and a 1-based rank (1 = strongest). Assets too short for a lookback get NaN."""
    rows: dict[str, dict] = {}
    for name, s in closes.items():
        s = s.dropna()
        rows[name] = {f'mom_{lb}': (np.log(s.iloc[-1] / s.iloc[-lb - 1])
                                    if len(s) > lb else np.nan) for lb in lookbacks}
    tab = pd.DataFrame(rows).T
    tab['score'] = tab.mean(axis=1)
    tab['rank'] = tab['score'].rank(ascending=False, method='min').astype('Int64')
    return tab.sort_values('rank')
