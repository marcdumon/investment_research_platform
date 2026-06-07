"""Analysis-page boundary: fetch price series from the panel, resample to the chosen
frequency, and orchestrate the pure `irp.analysis.stats` functions into one result.

Pages import only from here (services-only rule). No Dash, no figures — the page turns
this data into charts.
"""
import datetime
import logging
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from irp.analysis import stats as _st
from irp.panel.load import available_tickers, load_prices_wide
from irp.query.simfin import sector_map as _sector_map
from irp.ui.services import universe_service

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
