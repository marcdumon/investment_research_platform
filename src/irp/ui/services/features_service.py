"""Feature-engineering service: assemble a long ML panel from cached snapshots.

Page boundary for the `/features` page. Wraps the snapshot cache, the panel
forward-returns engine, the pure `irp.features.engineering` ops, and the
named-recipe store. Pages import only from here.
"""
import datetime
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from irp.core.config import config
from irp.factors.cache import snapshot as _snapshot
from irp.factors.registry import _all_factors as _all_factors
from irp.features import engineering as _eng
from irp.panel.returns import forward_returns_panel as _forward_returns
from irp.ui.services import universe_service

logger = logging.getLogger(__name__)

_FREQ_MAP = {'Q': 'QE', 'A': 'YE'}
_EXPORT_DIR = Path(config.data.root_dir) / 'feature_exports'


def available_columns() -> list[str]:
    """Base feature palette: the registered factor columns persisted per snapshot."""
    return [f.name for f in _all_factors()]


def precompute(start_year: int, end_year: int, variant: Literal['A', 'Q']) -> int:
    """Populate the snapshot cache for a variant over a year range (quarter-ends).

    Skips already-cached dates. Returns the number of new snapshots written.
    """
    start = datetime.date(int(start_year), 1, 1)
    end = datetime.date(int(end_year), 12, 31)
    return _snapshot.precompute_all(start, end, variants=[variant], freq='QE')


def _date_grid(start_year: int, end_year: int, freq: str) -> list[datetime.date]:
    pd_freq = _FREQ_MAP.get(freq, 'YE')
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    return [ts.date() for ts in pd.date_range(start, end, freq=pd_freq)]


def build_panel(
    start_year: int,
    end_year: int,
    freq: Literal['Q', 'A'],
    variant: Literal['A', 'Q'],
    steps: list[dict],
    label_cfg: dict,
    market: str | None = None,
    sector: str | None = None,
    watchlist: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the long (Date, Ticker) feature panel.

    Returns (panel, missing_dates). When any grid date has no cached snapshot,
    returns (empty df, [those dates as ISO strings]) WITHOUT computing inline —
    the caller surfaces a "run precompute_all first" message. This keeps the
    synchronous callback fast and avoids accidental multi-minute recomputes.
    """
    grid = _date_grid(start_year, end_year, freq)
    if not grid:
        return pd.DataFrame(), []

    tickers = universe_service._filter_tickers(market, sector, watchlist)

    snapshots: dict[datetime.date, pd.DataFrame] = {}
    missing: list[str] = []
    for d in grid:
        snap = _snapshot.load(d, variant)
        if snap is None:
            missing.append(d.isoformat())
            continue
        if tickers is not None:
            snap = snap[snap.index.isin(tickers)]
        snapshots[d] = snap

    # Hard-guard only when the cache is fully cold (nothing usable). When SOME
    # dates are cached, skip the uncached ones (e.g. market-holiday quarter-ends
    # that yield empty cross-sections and can never be cached) and build from the
    # rest — `missing` is returned as a non-blocking "skipped" warning.
    if not snapshots:
        return pd.DataFrame(), missing

    panel = _eng.assemble_panel(snapshots)
    if panel.empty:
        return panel, []

    sector_series = None
    feature_cols: list[str] = []
    for step in steps:
        if step.get('op') == 'norm' and step.get('method') == 'sector':
            if sector_series is None:
                from irp.query.simfin import sector_map
                sector_series = sector_map()
            step = {**step, 'sector': sector_series}
        panel = _eng.apply_step(panel, step)
        for c in _eng.step_output_cols(step):
            if c not in feature_cols:
                feature_cols.append(c)

    # Keep only the user-selected features (the 39 base columns are inputs, not
    # output): Date, Ticker, each step's produced column(s), then the label.
    keep = ['Date', 'Ticker'] + [c for c in feature_cols if c in panel.columns]
    panel = panel[keep]

    mode = label_cfg.get('mode', 'none')
    if mode != 'none':
        horizon = int(label_cfg.get('horizon_days', 252))
        fwd = _forward_returns(grid, horizon, tickers=tickers)
        panel = _eng.attach_label(
            panel, fwd, mode=mode, n_buckets=int(label_cfg.get('n_buckets', 5))
        )
    return panel, missing


def export_panel(df: pd.DataFrame, fmt: Literal['parquet', 'csv'], name: str = 'features') -> Path:
    """Write the panel to data/feature_exports/<name>_<timestamp>.<fmt>; return path."""
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    safe = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in name) or 'features'
    path = _EXPORT_DIR / f'{safe}_{ts}.{fmt}'
    if fmt == 'csv':
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path


# ── recipe passthroughs (keep page services-only) ─────────────────────

def list_recipes() -> pd.DataFrame:
    from irp.query import feature_recipes
    return feature_recipes.list_recipes()


def load_recipe(name: str) -> dict:
    from irp.query import feature_recipes
    return feature_recipes.load_recipe(name)


def save_recipe(name: str, spec: dict) -> None:
    from irp.query import feature_recipes
    feature_recipes.save_recipe(name, spec)


def delete_recipe(name: str) -> None:
    from irp.query import feature_recipes
    feature_recipes.delete_recipe(name)
