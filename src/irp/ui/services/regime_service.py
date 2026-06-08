"""Macro-regime boundary for the `/regime` page.

Assembles a cross-asset feature panel from the Stooq yield/equity/commodity series + a
EURUSD-derived USD proxy, classifies the regime two ways (rule-based risk score + Gaussian
HMM), and feeds the regime back into stock decisions: factor IC conditioned on regime, a
regime-gated factor backtest, and a cross-asset tactical-allocation table. All pure math
lives in `irp.analysis.regime`; this layer only sources data and caches.

Data note: yields (`10YUSY/2YUSY/3MUSY`), `^SPX`, `^CRY` come from the Stooq `prices` table
(deep history). `USDX` in the Yahoo panel starts only 2024, so the USD-strength feature uses
`1/EURUSD` from Stooq (history back to 1971). Crypto is dropped from the default macro set
(only short-history ETFs exist) so the HMM gets a clean 1995+ complete matrix.

Conditioning/gating use the **rule** labels: they are causal (expanding z-score) and cheap,
so the interactive paths avoid the slow expanding-HMM refit. The HMM is full-sample, for the
dashboard timeline only (in-sample; labeled as such).
"""
import datetime
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from irp.analysis import regime as _rg
from irp.features.composite import PRESETS
from irp.panel.load import load_prices_wide
from irp.query import stooq as _stooq
from irp.ui.services import analysis_service, backtest_service

logger = logging.getLogger(__name__)

# role -> (ticker, invert). Yields/equity/commodity + USD proxy all from Stooq `prices`.
_ROLE_SRC: dict[str, tuple[str, bool]] = {
    'y10': ('10YUSY', False), 'y2': ('2YUSY', False), 'y3m': ('3MUSY', False),
    'equity': ('^SPX', False), 'commod': ('^CRY', False),
    'usd': ('EURUSD', True),                 # invert -> USD strength
}
_EQUITY = '^SPX'

# Cross-asset proxies for the tactical table (Yahoo panel ETFs/indices).
_TACTICAL: dict[str, str] = {
    'US Equities': '^SPX', 'Nasdaq': '^NDX', 'Long Treasuries': 'TLT',
    'Mid Treasuries': 'IEF', 'Gold': 'GLD', 'Commodities': 'DBC',
    'HY Credit': 'HYG', 'Crypto': 'BTCO',
}

_PANEL_CACHE: dict[tuple, pd.DataFrame] = {}
_STOOQ_CACHE: dict[str, pd.Series] = {}
_CACHE_CAP = 4


@dataclass
class RegimeState:
    """Dashboard payload: classifiers + the series needed to draw them."""
    features: pd.DataFrame
    rule: pd.DataFrame                  # risk_score + label, daily
    hmm_labels: pd.Series               # full-sample HMM state per day (display)
    hmm_transition: pd.DataFrame
    hmm_means: pd.DataFrame
    equity: pd.Series                   # ^SPX close for the timeline overlay
    contrib: pd.Series                  # latest per-feature risk-on contribution
    n_states: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConditionedFactors:
    signal: str
    table: pd.DataFrame                 # index regime, cols mean_ic/icir/n
    n_dates: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class GatedBacktest:
    signal: str
    allowed: list[str]
    result: dict                        # gated/base period + cumret + sharpe + maxdd
    warnings: list[str] = field(default_factory=list)


@dataclass
class MarkovResult:
    """Single-asset Markov regime ('hedge-fund method') payload."""
    ticker: str
    states: pd.Series
    transition: pd.DataFrame
    current_state: str
    signal: float                       # P(bull)−P(bear) for the current state
    n_step: pd.DataFrame                # distribution 1..N days ahead
    stationary: pd.Series
    hmm_states: pd.Series               # HMM bull/sideways/bear, aligned (empty if off)
    hmm_agree: float                    # fraction where rule & HMM agree (nan if off)
    equity: pd.Series
    backtest: dict
    lookback: int
    threshold: float
    overlapping: bool
    warnings: list[str] = field(default_factory=list)


_STATE_FROM_INT = {0: 'bear', 1: 'sideways', 2: 'bull'}


def _stooq_close(ticker: str) -> pd.Series:
    """Daily close from the Stooq `prices` table, one row per Date (deduped on Src)."""
    if ticker in _STOOQ_CACHE:
        return _STOOQ_CACHE[ticker]
    df = _stooq.prices(ticker)
    if df.empty:
        s = pd.Series(dtype=float)
    else:
        df = df.sort_values('Date').drop_duplicates('Date', keep='last')
        s = pd.Series(df['C'].to_numpy(float), index=pd.to_datetime(df['Date']))
    _STOOQ_CACHE[ticker] = s
    return s


def _panel_close(ticker: str) -> pd.Series:
    """Daily close from the Yahoo price panel (for the tactical ETF proxies)."""
    p = load_prices_wide('Close')
    i = p.ticker_to_idx.get(ticker)
    if i is None:
        return pd.Series(dtype=float)
    return pd.Series(p.values[:, i], index=pd.to_datetime(p.dates)).dropna()


def _role_closes(start, end) -> dict[str, pd.Series]:
    closes: dict[str, pd.Series] = {}
    for role, (ticker, invert) in _ROLE_SRC.items():
        s = _stooq_close(ticker)
        if s.empty:
            continue
        if start is not None:
            s = s[s.index >= pd.Timestamp(start)]
        if end is not None:
            s = s[s.index <= pd.Timestamp(end)]
        closes[role] = (1.0 / s) if invert else s
    return closes


def feature_panel(start: datetime.date | None, end: datetime.date | None,
                  standardize: bool = True) -> pd.DataFrame:
    """Standardized cross-asset macro feature panel over the window (cached)."""
    key = (start, end, standardize)
    if key in _PANEL_CACHE:
        return _PANEL_CACHE[key]
    feats = _rg.build_feature_panel(_role_closes(start, end), standardize=standardize)
    while len(_PANEL_CACHE) >= _CACHE_CAP:
        _PANEL_CACHE.pop(next(iter(_PANEL_CACHE)))
    _PANEL_CACHE[key] = feats
    return feats


def dashboard(start: datetime.date | None, end: datetime.date | None,
              n_states: int = 3) -> RegimeState:
    """Rule + full-sample HMM classification + the ^SPX overlay for the timeline.

    Classifiers fit on FULL history up to `end` (so the expanding z-baseline and HMM are
    stable regardless of the chosen window and there is no per-preset burn-in); the series
    are then sliced to `[start, end]` for display only."""
    warnings: list[str] = []
    feats_full = feature_panel(None, end)
    if feats_full.dropna().empty:
        warnings.append('no macro feature history — widen the period')
        return RegimeState(pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=int),
                           pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float),
                           pd.Series(dtype=float), n_states, warnings)
    rule_full = _rg.rule_regime(feats_full)
    fit = _rg.hmm_regime(feats_full.dropna(), n_states=n_states, mode='full')
    contrib = _rg.feature_contributions(feats_full)
    eq = _stooq_close(_EQUITY)
    if end is not None:
        eq = eq[eq.index <= pd.Timestamp(end)]
    lo = pd.Timestamp(start) if start is not None else feats_full.index.min()
    return RegimeState(
        feats_full[feats_full.index >= lo], rule_full[rule_full.index >= lo],
        fit.labels[fit.labels.index >= lo], fit.transition, fit.state_means,
        eq[eq.index >= lo], contrib, n_states, warnings)


def current_regime(end: datetime.date | None = None) -> dict:
    """Light current-state read for the /today banner: latest rule label + risk score +
    top feature drivers. Fast — full-history rule classifier, no HMM."""
    feats = feature_panel(None, end)
    rule = _rg.rule_regime(feats)
    valid = rule.dropna()
    if valid.empty:
        return {'label': 'unknown', 'score': np.nan, 'drivers': pd.Series(dtype=float),
                'as_of': None}
    last = valid.iloc[-1]
    contrib = _rg.feature_contributions(feats)
    return {'label': str(last['label']), 'score': float(last['risk_score']),
            'drivers': contrib, 'as_of': valid.index[-1].date()}


def _rule_labels(end) -> pd.Series:
    """Daily causal rule regime labels over FULL history up to `end` (PIT — expanding
    z-score). Full history so conditioning at a rebalance date uses everything knowable
    then, not a window-truncated baseline."""
    return _rg.rule_regime(feature_panel(None, end))['label']


def conditioned_factors(
    signal: str, start: datetime.date, end: datetime.date, horizon: int = 63,
    variant: str = 'A', freq: str = 'Q', market: str | None = None,
    sector: str | None = None, watchlist: str | None = None,
) -> ConditionedFactors:
    """Factor IC grouped by the (causal rule) regime that held at each rebalance date.

    `signal` is a factor name or a `composite.PRESETS` key. Answers 'which regime is this
    signal's edge concentrated in'.
    """
    warnings: list[str] = []
    if signal in PRESETS:
        br = backtest_service._run_composite(
            PRESETS[signal], horizon, start, end, variant=variant, freq=freq,
            market=market, sector=sector, watchlist=watchlist)
    else:
        br = backtest_service._run_factor(
            signal, horizon, start, end, variant=variant, freq=freq,
            market=market, sector=sector, watchlist=watchlist)
    ic = br.ic_series.dropna() if br.ic_series is not None else pd.Series(dtype=float)
    if ic.empty:
        warnings.append('no IC series — warm the factor cache (precompute_all) for this period')
        return ConditionedFactors(signal, pd.DataFrame(), 0, warnings)
    labels = _rg.align_to_dates(_rule_labels(end), ic.index)
    table = _rg.regime_conditioned_ic(ic, labels)
    return ConditionedFactors(signal, table, int(len(ic)), warnings)


def gated_backtest(
    signal: str, allowed: list[str], start: datetime.date, end: datetime.date,
    horizon: int = 63, variant: str = 'A', freq: str = 'Q', market: str | None = None,
    sector: str | None = None, watchlist: str | None = None,
) -> GatedBacktest:
    """Long-short factor return taken only in `allowed` regimes vs always-on."""
    warnings: list[str] = []
    if signal in PRESETS:
        br = backtest_service._run_composite(
            PRESETS[signal], horizon, start, end, variant=variant, freq=freq,
            market=market, sector=sector, watchlist=watchlist)
    else:
        br = backtest_service._run_factor(
            signal, horizon, start, end, variant=variant, freq=freq,
            market=market, sector=sector, watchlist=watchlist)
    ls = br.ls_cumret.dropna() if br.ls_cumret is not None else pd.Series(dtype=float)
    if ls.empty or len(ls) < 4:
        warnings.append('no long-short return series for this period/signal')
        return GatedBacktest(signal, allowed, {}, warnings)
    period = ls.diff().dropna()
    labels = _rg.align_to_dates(_rule_labels(end), period.index)
    ppy = max(1, round(252 / horizon))
    res = _rg.gated_returns(period, labels, set(allowed), ppy=ppy)
    return GatedBacktest(signal, allowed, res, warnings)


def tactical(start: datetime.date | None, end: datetime.date | None,
             lookbacks=(63, 126, 252)) -> pd.DataFrame:
    """Cross-asset trend/momentum table over the asset-class proxies (latest as of `end`)."""
    closes: dict[str, pd.Series] = {}
    for name, ticker in _TACTICAL.items():
        s = _panel_close(ticker)
        if s.empty:
            s = _stooq_close(ticker)
        if s.empty:
            continue
        if start is not None:
            s = s[s.index >= pd.Timestamp(start)]
        if end is not None:
            s = s[s.index <= pd.Timestamp(end)]
        closes[name] = s
    if not closes:
        return pd.DataFrame()
    return _rg.tactical_table(closes, lookbacks=lookbacks)


def _any_close(ticker: str, start, end) -> pd.Series:
    """Close for any tradeable ticker: Yahoo panel first, Stooq fallback, windowed."""
    s = analysis_service._close_series(ticker, start, end)
    if s.empty:
        s = _stooq_close(ticker)
        if start is not None:
            s = s[s.index >= pd.Timestamp(start)]
        if end is not None:
            s = s[s.index <= pd.Timestamp(end)]
    return s


def _hmm_single_asset(close: pd.Series, vol_win: int = 20) -> pd.Series:
    """HMM bull/sideways/bear labels from one asset's return + vol (states ordered by
    mean return → 0=bear, 2=bull)."""
    ret = np.log(close / close.shift(1))
    feats = pd.concat([ret.rename('ret'), ret.rolling(vol_win).std().rename('vol')],
                      axis=1).dropna()
    if len(feats) < 50:
        return pd.Series(dtype=object)
    fit = _rg.hmm_regime(feats, n_states=3, mode='full')
    return fit.labels.map(_STATE_FROM_INT)


def markov_analysis(ticker: str, start: datetime.date | None, end: datetime.date | None,
                    lookback: int = 20, threshold: float = 0.05, overlapping: bool = True,
                    hmm_overlay: bool = True, n_step: int = 20,
                    horizon: int = 5) -> MarkovResult:
    """Full single-asset Markov regime: states, transition matrix, n-step forecast,
    stationary distribution, current signal, optional HMM agreement, walk-forward backtest."""
    warnings: list[str] = []
    close = _any_close(ticker, start, end)
    empty = MarkovResult(ticker, pd.Series(dtype=object), pd.DataFrame(), '—', np.nan,
                         pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=object),
                         np.nan, pd.Series(dtype=float), {}, lookback, threshold,
                         overlapping, warnings)
    if close.empty or len(close) < lookback + 30:
        warnings.append(f'not enough price history for {ticker}')
        return empty
    states = _rg.return_states(close, lookback, threshold, overlapping)
    if states.empty:
        warnings.append('no states computed')
        return empty
    P = _rg.transition_matrix(states)
    current = str(states.iloc[-1])
    signal = _rg.directional_signal(P, current)
    nstep = _rg.n_step_distribution(P, current, steps=n_step)
    stat = _rg.stationary_distribution(P)
    bt = _rg.markov_backtest(close, lookback, threshold, overlapping, horizon=horizon)

    hmm_states = pd.Series(dtype=object)
    agree = np.nan
    if hmm_overlay:
        hmm_states = _hmm_single_asset(close)
        if not hmm_states.empty:
            j = pd.concat([states.rename('rule'),
                           hmm_states.reindex(states.index, method='ffill').rename('hmm')],
                          axis=1).dropna()
            if len(j):
                agree = float((j['rule'] == j['hmm']).mean())
        else:
            warnings.append('HMM overlay skipped — too little history')
    return MarkovResult(ticker, states, P, current, signal, nstep, stat, hmm_states, agree,
                        close, bt, lookback, threshold, overlapping, warnings)


def signal_options() -> list[dict]:
    """Factor + composite-preset choices for the conditioning/gating pickers."""
    from irp.ui.factor_meta import FACTOR_OPTIONS
    presets = [{'label': f'Composite: {k}', 'value': k} for k in PRESETS]
    return presets + list(FACTOR_OPTIONS)


def instrument_options() -> list[dict]:
    """All panel tickers, for the single-asset Markov picker."""
    return [{'label': t, 'value': t} for t in analysis_service._available_instruments()]
