"""Analysis-page boundary: fetch price series from the panel, resample to the chosen
frequency, and orchestrate the pure `irp.analysis.stats` functions into one result.

Pages import only from here (services-only rule). No Dash, no figures — the page turns
this data into charts.
"""
import datetime
import logging
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from irp.analysis import events as _ev, pairs as _pairs, stats as _st
from irp.panel.load import available_tickers, load_prices_wide
from irp.query.simfin import fundamentals as _fundamentals, sector_map as _sector_map
from irp.ui.services import price_service, universe_service

logger = logging.getLogger(__name__)

_ROLL_VOL_WINDOW = {'D': 63, 'W': 13, 'M': 6}      # ~quarter at each frequency
_ROLL_BETA_WINDOW = {'D': 126, 'W': 26, 'M': 12}   # ~6-12 months

# Universe Market values that are index instruments.
_INDEX_MARKETS = {'indices', 'stooq stocks indices'}

# Best-effort human names for the index tickers we carry (data has no name column).
_INDEX_NAMES = {
    '^AEX': 'AEX (Amsterdam)', '^BUX': 'BUX (Budapest)', '^DJI': 'Dow Jones Industrial',
    '^HSI': 'Hang Seng', '^IBEX': 'IBEX 35 (Spain)', '^IPC': 'IPC (Mexico)',
    '^IPSA': 'IPSA (Chile)', '^MDAX': 'MDAX (Germany)', '^NDX': 'Nasdaq 100',
    '^NZ50': 'NZX 50', '^OMXC25': 'OMX Copenhagen 25', '^OSEAX': 'Oslo All-Share',
    '^SPX': 'S&P 500', '^STI': 'Straits Times (Singapore)', '^XU100': 'BIST 100 (Turkey)',
}


@dataclass
class AnalysisResult:
    ticker: str
    benchmark: str | None
    freq: str
    ppy: int
    summary: dict
    hist: tuple
    qq: tuple
    cumulative: pd.Series
    drawdown: pd.Series
    rolling_vol: pd.Series
    acf: tuple
    adf: tuple
    market: dict | None = None
    rolling_beta: pd.Series | None = None
    resid_qq: tuple | None = None
    resid_acf: tuple | None = None
    peers_corr: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)


def _available_instruments() -> list[str]:
    """All tickers in the price panel (any instrument type). Uses the cheap
    Ticker-only read so opening the page doesn't trigger the dense-matrix build."""
    return available_tickers()


@lru_cache(maxsize=1)
def _benchmark_options() -> list[dict]:
    """Index instruments present in the price panel, as dropdown options. Label is the
    best-effort human name + ticker (the data carries no name column)."""
    panel = set(available_tickers())
    uni = universe_service._get_universe()
    idx = uni[uni['Market'].isin(_INDEX_MARKETS)]
    opts = []
    for t in sorted(idx['Ticker']):
        if t in panel:
            name = _INDEX_NAMES.get(t)
            opts.append({'label': f'{name} ({t})' if name else t, 'value': t})
    return opts


@lru_cache(maxsize=1)
def _sector_options() -> list[dict]:
    """Sectors available for filtering the peers list."""
    return [{'label': s, 'value': s} for s in universe_service._get_sectors()]


def _ticker_sector(ticker: str | None) -> str | None:
    """Sector the instrument belongs to (None if unmapped)."""
    if not ticker:
        return None
    val = _sector_map().get(ticker)
    return str(val) if val is not None and pd.notna(val) else None


def _peers_options(sector: str | None = None) -> list[dict]:
    """Peer-instrument options, optionally restricted to one sector. Without a sector
    the full panel is offered; with one, only that sector's tickers that exist in the
    panel (keeps the list short and relevant)."""
    panel = set(available_tickers())
    if not sector:
        return [{'label': t, 'value': t} for t in sorted(panel)]
    sm = _sector_map()
    tickers = [t for t in sm[sm == sector].index if t in panel]
    return [{'label': t, 'value': t} for t in sorted(tickers)]


def _close_series(ticker: str, start: datetime.date | None, end: datetime.date | None) -> pd.Series:
    """Close-price Series (Timestamp index) for one ticker from the panel, windowed."""
    panel = load_prices_wide('Close')
    idx = panel.ticker_to_idx.get(ticker)
    if idx is None:
        return pd.Series(dtype=float)
    s = pd.Series(panel.values[:, idx], index=pd.to_datetime(panel.dates)).dropna()
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    return s


def _log_returns(ticker: str, start, end, freq: str) -> pd.Series:
    """Resampled (period-end) log returns for one ticker."""
    close = _close_series(ticker, start, end)
    if close.empty:
        return pd.Series(dtype=float)
    return _st.to_log_returns(_st.resample_close(close, freq))


def _peers_corr(tickers: list[str], start, end, freq: str) -> pd.DataFrame | None:
    """Pairwise log-return correlation of the focal ticker + peers at the chosen freq."""
    cols = {}
    for t in tickers:
        r = _log_returns(t, start, end, freq)
        if not r.empty:
            cols[t] = r
    if len(cols) < 2:
        return None
    frame = pd.DataFrame(cols).dropna(how='all')
    if frame.shape[0] < 5:
        return None
    return frame.corr(method='pearson')


def _analyze(
    ticker: str,
    benchmark: str | None,
    peers: list[str] | None,
    start: datetime.date | None,
    end: datetime.date | None,
    freq: str,
) -> AnalysisResult:
    """Full return-statistics report for `ticker`, optionally vs `benchmark` + `peers`."""
    ppy = _st.PERIODS_PER_YEAR.get(freq, 252)
    warnings: list[str] = []

    rets = _log_returns(ticker, start, end, freq)
    if rets.size < 20:
        warnings.append(f'only {rets.size} return obs for {ticker} at {freq} — stats unreliable')

    res = AnalysisResult(
        ticker=ticker, benchmark=benchmark, freq=freq, ppy=ppy,
        summary=_st.summary_stats(rets, ppy),
        hist=_st.histogram_normal(rets),
        qq=_st.qq_points(rets),
        cumulative=_st.cumulative(rets),
        drawdown=_st.drawdown(rets),
        rolling_vol=_st.rolling_vol(rets, _ROLL_VOL_WINDOW.get(freq, 63), ppy),
        acf=_st.autocorr(rets, nlags=min(40, max(1, rets.size - 1))),
        adf=_st.adf(rets),
        warnings=warnings,
    )

    if benchmark and benchmark != ticker:
        bench_rets = _log_returns(benchmark, start, end, freq)
        mm = _st.market_model(rets, bench_rets, ppy)
        res.market = mm
        res.rolling_beta = _st.rolling_beta(rets, bench_rets, _ROLL_BETA_WINDOW.get(freq, 126))
        resid = mm.get('residuals')
        if resid is not None and len(resid) >= 20:
            res.resid_qq = _st.qq_points(resid)
            res.resid_acf = _st.autocorr(resid, nlags=min(40, max(1, len(resid) - 1)))
        elif mm['n'] < 20:
            warnings.append(f'only {mm["n"]} overlapping obs with {benchmark} — beta unreliable')

    peer_set = [ticker] + [p for p in (peers or []) if p and p != ticker]
    if len(peer_set) > 1:
        res.peers_corr = _peers_corr(peer_set, start, end, freq)
        if res.peers_corr is None:
            warnings.append('not enough overlapping history for peers correlation')

    return res


# ── pairs / cointegration ──────────────────────────────────────────────

@dataclass
class PairResult:
    a: str
    b: str
    eg: dict                       # engle_granger output (coint_t, pvalue, hedge_ratio, spread, …)
    zscore: pd.Series
    half_life: float
    leadlag: tuple                 # (lags, xcorr, best_lag)
    overlay: pd.DataFrame          # a_norm / b_norm rebased to 100
    rolling_corr: pd.Series
    log_a: pd.Series               # aligned log prices (for the hedge scatter)
    log_b: pd.Series
    n: int
    warnings: list[str] = field(default_factory=list)


def _pair_analysis(a: str, b: str, start, end) -> PairResult | None:
    """Cointegration / stat-arb diagnostics for instruments A and B on log prices.
    Reuses `_close_series`; returns None when either series is missing."""
    ca = _close_series(a, start, end)
    cb = _close_series(b, start, end)
    if ca.empty or cb.empty:
        return None
    j = pd.concat([ca.rename('a'), cb.rename('b')], axis=1, join='inner').dropna()
    warnings: list[str] = []
    if len(j) < 60:
        warnings.append(f'only {len(j)} overlapping days — pair stats unreliable')
    if len(j) < 20:
        return PairResult(a, b, {}, pd.Series(dtype=float), float('nan'), ([], [], 0),
                          pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float),
                          pd.Series(dtype=float), len(j), warnings)

    la = np.log(j['a'])
    lb = np.log(j['b'])
    eg = _pairs.engle_granger(la, lb)
    spread = eg['spread']
    z = _pairs.spread_zscore(spread)
    hl = _pairs.half_life(spread)
    ra, rb = la.diff(), lb.diff()
    leadlag = _pairs.lead_lag(ra, rb, max_lag=10)
    overlay = pd.DataFrame({
        'a_norm': j['a'] / j['a'].iloc[0] * 100.0,
        'b_norm': j['b'] / j['b'].iloc[0] * 100.0,
    })
    rolling_corr = ra.rolling(63).corr(rb)
    return PairResult(a, b, eg, z, hl, leadlag, overlay, rolling_corr, la, lb, len(j), warnings)


# ── event study & seasonality ──────────────────────────────────────────

@dataclass
class EventResult:
    ticker: str
    event_type: str
    method: str
    aar: pd.Series
    car: pd.Series
    n: int
    car_end: float
    tstat: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class SeasonalityResult:
    ticker: str
    monthly: pd.Series
    dow: pd.Series
    n: int


def _event_dates(ticker: str, event_type: str) -> list:
    """Event dates for the instrument: earnings = SimFin Publish Date (quarterly);
    dividends / splits = Yahoo action dates."""
    if event_type == 'earnings':
        df = _fundamentals(ticker, 'income', variant='Q')
        if df.empty or 'Publish Date' not in df.columns:
            return []
        return list(pd.to_datetime(df['Publish Date'], errors='coerce').dropna())
    if event_type == 'dividends':
        df = price_service._get_dividends(ticker)
    elif event_type == 'splits':
        df = price_service._get_splits(ticker)
    else:
        return []
    if df.empty or 'Date' not in df.columns:
        return []
    return list(pd.to_datetime(df['Date'], errors='coerce').dropna())


def _event_study(ticker: str, event_type: str, method: str, benchmark: str | None,
                 start, end, pre: int = 10, post: int = 10) -> EventResult:
    """CAR/AAR of `ticker` around its `event_type` dates under the chosen abnormal-return
    `method` (market/mean/raw)."""
    warnings: list[str] = []
    rets = _log_returns(ticker, start, end, 'D')
    if rets.empty:
        warnings.append(f'no price history for {ticker}')
        return EventResult(ticker, event_type, method, pd.Series(dtype=float),
                           pd.Series(dtype=float), 0, float('nan'), float('nan'), warnings)
    lo, hi = rets.index.min(), rets.index.max()
    dates = [d for d in _event_dates(ticker, event_type) if lo <= pd.Timestamp(d) <= hi]
    if not dates:
        warnings.append(f'no {event_type} events for {ticker} in this period')
        return EventResult(ticker, event_type, method, pd.Series(dtype=float),
                           pd.Series(dtype=float), 0, float('nan'), float('nan'), warnings)

    if method == 'mean':
        # textbook event study: each event's 'normal' = mean over a pre-event window
        matrix = _ev.mean_adjusted_matrix(rets, dates, pre, post)
    else:
        market = _log_returns(benchmark or '^SPX', start, end, 'D') if method == 'market' else None
        ar = _ev.abnormal_returns(rets, market, method=method)
        matrix = _ev.event_window_matrix(ar, dates, pre, post)
    res = _ev.aar_car(matrix)
    if res['n'] < 3:
        warnings.append(f'only {res["n"]} usable {event_type} events — AAR/CAR noisy')
    return EventResult(ticker, event_type, method, res['aar'], res['car'], res['n'],
                       res['car_end'], res['tstat'], warnings)


def _seasonality(ticker: str, start, end) -> SeasonalityResult:
    """Month-of-year and day-of-week average daily returns for the instrument."""
    rets = _log_returns(ticker, start, end, 'D')
    return SeasonalityResult(ticker, _ev.monthly_seasonality(rets),
                             _ev.dow_seasonality(rets), int(rets.size))
