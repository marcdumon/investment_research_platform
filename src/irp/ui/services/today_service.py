"""`/today` decision-surface boundary.

Composes the regime read, the regime → playbook map, and a composite ranking of the latest
cached cross-section into one "what do I do today" payload. Thin: all analytics live in
`regime_service`, `factors_service`, and `features.composite`. No Dash, no figures.
"""
import datetime
import logging
from dataclasses import dataclass, field

import pandas as pd

from irp.factors.cache import snapshot
from irp.features.composite import PRESETS, build_composite
from irp.ui.services import factors_service, regime_service, watchlist_service

logger = logging.getLogger(__name__)

# Regime → recommended composite preset + plain-language stance. Transparent, hand-set.
_REGIME_PLAYBOOK: dict[str, dict[str, str]] = {
    'risk_on':  {'preset': 'momentum',  'stance': 'Lean into trend; full gross'},
    'neutral':  {'preset': 'composite', 'stance': 'Balanced; normal gross'},
    'risk_off': {'preset': 'quality',   'stance': 'Defensive; tilt to quality, reduce gross'},
    'unknown':  {'preset': 'composite', 'stance': 'Insufficient regime history — neutral'},
}

_REGIME_COLOR = {'risk_on': '#4ec94e', 'neutral': '#9aa0a6', 'risk_off': '#e45756',
                 'unknown': '#9aa0a6'}


@dataclass
class TodayResult:
    regime: dict                        # label, score, drivers, as_of
    playbook: dict                      # preset, stance
    preset: str                         # composite actually used for ranking
    as_of: datetime.date | None         # cross-section date
    top: pd.DataFrame                   # ranked names
    watchlist: pd.DataFrame             # watchlist standing (empty if none)
    variant: str
    warnings: list[str] = field(default_factory=list)


def playbook_preset(label: str) -> str:
    return _REGIME_PLAYBOOK.get(label, _REGIME_PLAYBOOK['unknown'])['preset']


def regime_color(label: str) -> str:
    return _REGIME_COLOR.get(label, _REGIME_COLOR['unknown'])


def _apply_universe_floor(df: pd.DataFrame, min_mktcap: float) -> pd.DataFrame:
    """Restrict to an investable universe by market cap ($B). Names without a market cap
    are dropped (can't be sized). No `mktcap` column → unchanged."""
    if min_mktcap <= 0 or 'mktcap' not in df.columns:
        return df
    return df[df['mktcap'] >= min_mktcap]


def _rank_by_composite(df: pd.DataFrame, weights: dict[str, float], n: int = 20) -> pd.DataFrame:
    """Top-`n` names by the composite of the available weighted factors.

    Weights whose factor columns are absent from `df` are dropped (a cross-section may not
    carry every factor); if none remain, returns an empty frame rather than raising. Uses
    **rank** normalization so scores spread across [-0.5, +0.5] (no z-score clip ties at the
    top). Output carries a leading `Rank` + a `score` column, NaN-score rows dropped."""
    avail = {k: v for k, v in weights.items() if k in df.columns}
    if not avail or df.empty:
        return pd.DataFrame()
    score = build_composite(df, avail, normalize='rank')
    out = df.copy()
    out['score'] = score
    out = out.dropna(subset=['score']).sort_values('score', ascending=False).head(n)
    out.insert(0, 'Rank', range(1, len(out) + 1))
    return out


def _display_cols(preset: str) -> list[str]:
    """Columns to surface in the top-names table for a preset."""
    base = ['Rank', 'Company Name', 'Sector', 'mktcap', 'score']
    return base + list(PRESETS[preset].keys())


def dashboard(variant: str = 'A', market: str | None = None, sector: str | None = None,
              watchlist: str | None = None, top_n: int = 20, preset: str | None = None,
              min_mktcap: float = 1.0) -> TodayResult:
    """Assemble the /today payload: regime + playbook + top names + watchlist standing.

    `min_mktcap` ($B, default $1B) restricts the ranked universe to investable names — a
    naive factor rank otherwise surfaces distressed micro-caps with blown-up ratios."""
    warnings: list[str] = []
    regime = regime_service.current_regime()
    label = regime['label']
    play = _REGIME_PLAYBOOK.get(label, _REGIME_PLAYBOOK['unknown'])
    use_preset = preset or play['preset']

    as_of = snapshot.latest_cached(variant)
    empty = TodayResult(regime, play, use_preset, as_of, pd.DataFrame(), pd.DataFrame(),
                        variant, warnings)
    if as_of is None:
        warnings.append(f'no cached cross-section for variant {variant} — run Precompute on /features')
        return empty
    if regime.get('as_of') and as_of and str(regime['as_of']) != str(as_of):
        warnings.append(f'regime is current ({regime["as_of"]}) but names use the {as_of} '
                        f'cross-section cache — names may be stale')

    xs = factors_service._load_cross_section(as_of, variant, market=market, sector=sector,
                                             watchlist=watchlist)
    if xs.empty:
        warnings.append('cross-section is empty for these filters')
        return empty
    xs = _apply_universe_floor(xs, min_mktcap)
    if xs.empty:
        warnings.append(f'no names above ${min_mktcap:.0f}B market cap for these filters')
        return empty
    top = _rank_by_composite(xs, PRESETS[use_preset], n=top_n)
    keep = [c for c in _display_cols(use_preset) if c in top.columns]
    top_view = top[keep] if not top.empty else top

    wl = pd.DataFrame()
    if watchlist:
        wl = _watchlist_panel(as_of, variant, use_preset, watchlist)
    return TodayResult(regime, play, use_preset, as_of, top_view, wl, variant, warnings)


def _watchlist_panel(as_of: datetime.date, variant: str, preset: str,
                     name: str) -> pd.DataFrame:
    """Watchlist names with composite score + universe percentile + momentum."""
    full = factors_service._load_cross_section(as_of, variant)
    if full.empty:
        return pd.DataFrame()
    avail = {k: v for k, v in PRESETS[preset].items() if k in full.columns}
    if not avail:
        return pd.DataFrame()
    score = build_composite(full, avail)
    pct = score.rank(pct=True)
    tickers = [t for t in watchlist_service.load_watchlist(name) if t in full.index]
    if not tickers:
        return pd.DataFrame()
    out = pd.DataFrame({
        'Company Name': full.reindex(tickers).get('Company Name'),
        'Sector': full.reindex(tickers).get('Sector'),
        'score': score.reindex(tickers),
        'percentile': (pct.reindex(tickers) * 100).round(0),
        'mom_12_1': full.reindex(tickers).get('mom_12_1'),
    }, index=tickers)
    return out.sort_values('score', ascending=False)


def watchlist_options() -> list[dict]:
    wl = watchlist_service.list_watchlists()
    names = wl['name'].tolist() if not wl.empty and 'name' in wl.columns else []
    return [{'label': n, 'value': n} for n in names]


def sector_options() -> list[dict]:
    from irp.ui.services import universe_service
    return [{'label': s, 'value': s} for s in universe_service._get_sectors()]


def preset_options() -> list[dict]:
    return [{'label': 'Auto (regime)', 'value': 'auto'}] + \
        [{'label': k.title(), 'value': k} for k in PRESETS]
