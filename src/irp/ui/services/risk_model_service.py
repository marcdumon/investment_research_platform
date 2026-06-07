"""Multi-factor risk-model boundary for the `/analysis` Factor-model section.

Assembles a systematic factor-return panel from the existing backtest infra (the
long-short return of each style composite + the equal-weight universe return), pulls the
instrument's matching period returns from the price panel, and runs the pure
`irp.analysis.risk_model` regression. The factor-return panel is instrument-independent,
so it is cached and reused across tickers — only the final regression is per-instrument.
"""
import datetime
import logging
from dataclasses import dataclass, field

import pandas as pd

from irp.analysis import risk_model as _rm
from irp.features.composite import PRESETS
from irp.panel.returns import forward_returns_panel
from irp.ui.services import backtest_service

logger = logging.getLogger(__name__)

# Style factors built as long-short composites; 'market' is a real index return.
_STYLES = ('value', 'quality', 'momentum')
_FACTORS = ('market', *_STYLES)
_DEFAULT_MARKET = '^SPX'    # used when no benchmark index is selected

# instrument-independent STYLE-return panels, keyed by (start, end, freq, horizon, universe).
# The market factor is a cheap per-call index lookup, so it is NOT part of this cache.
_FACTOR_CACHE: dict[tuple, pd.DataFrame] = {}
_CACHE_CAP = 8


@dataclass
class RiskModelResult:
    ticker: str
    factors: list[str]
    regression: dict
    rolling_exposures: pd.DataFrame
    return_contrib: pd.Series
    risk_contrib: pd.Series
    factor_corr: pd.DataFrame
    freq: str
    horizon: int
    n: int
    warnings: list[str] = field(default_factory=list)


def _style_returns(
    start: datetime.date, end: datetime.date, freq: str, horizon: int,
    market_filter: str | None, sector: str | None, watchlist: str | None,
) -> pd.DataFrame:
    """Date × {value, quality, momentum} per-period long-short style returns from the
    backtest composites. Cached (slow to build); instrument- and benchmark-independent."""
    key = (start, end, freq, horizon, market_filter, sector, watchlist)
    if key in _FACTOR_CACHE:
        return _FACTOR_CACHE[key]

    cols: dict[str, pd.Series] = {}
    for style in _STYLES:
        br = backtest_service._run_composite(
            PRESETS[style], horizon, start, end, variant='A', freq=freq,
            market=market_filter, sector=sector, watchlist=watchlist)
        if br.ls_cumret is None or br.ls_cumret.empty:
            continue
        cols[style] = br.ls_cumret.diff()

    frame = pd.DataFrame(cols).dropna(how='all')
    frame = frame.reindex(columns=[c for c in _STYLES if c in frame.columns])
    while len(_FACTOR_CACHE) >= _CACHE_CAP:
        _FACTOR_CACHE.pop(next(iter(_FACTOR_CACHE)))
    _FACTOR_CACHE[key] = frame
    return frame


def _instrument_returns(ticker: str, index: pd.Index, horizon: int) -> pd.Series:
    """Instrument forward returns at the factor rebalance dates, on the factor index."""
    dates = [pd.Timestamp(d).date() for d in index]
    fwd = forward_returns_panel(dates, horizon, [ticker])
    if fwd.empty:
        return pd.Series(dtype=float)
    s = fwd[fwd['Ticker'] == ticker].copy()
    s.index = pd.to_datetime(s['Date'])
    return s['fwd_ret'].reindex(index)


def _risk_model(
    ticker: str, start: datetime.date, end: datetime.date, freq: str = 'Q',
    horizon: int = 63, benchmark: str | None = None, market_filter: str | None = None,
    sector: str | None = None, watchlist: str | None = None,
) -> RiskModelResult:
    """Full multi-factor risk model for one instrument over the period. The market
    factor is the chosen `benchmark` index (default `^SPX`); style factors are the
    cached long-short composites."""
    ppy = max(1, round(252 / horizon))
    warnings: list[str] = []
    styles = _style_returns(start, end, freq, horizon, market_filter, sector, watchlist)
    empty = RiskModelResult(ticker, list(_FACTORS), {}, pd.DataFrame(),
                            pd.Series(dtype=float), pd.Series(dtype=float),
                            pd.DataFrame(), freq, horizon, 0, warnings)
    if styles.empty:
        warnings.append('no factor returns — warm the factor cache (precompute_all) for this period')
        return empty

    # Market factor = the chosen benchmark index return over the same horizon/dates.
    mkt_ticker = benchmark or _DEFAULT_MARKET
    market_ret = _instrument_returns(mkt_ticker, styles.index, horizon)
    if market_ret.notna().sum() < 2:
        warnings.append(f'market index {mkt_ticker} has no usable history — using style factors only')
        X = styles
    else:
        X = styles.copy()
        X.insert(0, 'market', market_ret)

    y = _instrument_returns(ticker, X.index, horizon)
    aligned = pd.concat([y.rename('y'), X], axis=1, join='inner').dropna()
    n = int(len(aligned))
    if n <= len(X.columns) + 1:
        warnings.append(f'only {n} overlapping periods — too few for a {len(X.columns)}-factor fit')
        return empty
    if n < 12:
        warnings.append(f'thin sample ({n} quarterly periods) — exposures are noisy; use a longer period')

    reg = _rm.factor_regression(y, X, ppy=ppy)
    window = min(max(8, n // 3), n)
    roll = _rm.rolling_exposures(y, X, window=window)
    contrib = _rm.return_contributions(y, X, reg['betas'], reg['alpha'] / ppy)
    risk = _rm.risk_contributions(X, reg['betas'], reg['resid'])
    corr = X.corr()
    return RiskModelResult(ticker, list(X.columns), reg, roll, contrib, risk, corr,
                           freq, horizon, n, warnings)
