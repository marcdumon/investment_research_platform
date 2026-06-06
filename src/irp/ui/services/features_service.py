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
from irp.factors.registry import all_factors
from irp.features import engineering as _eng
from irp.panel.returns import forward_returns_panel as _forward_returns, price_volume_panel as _price_volume
from irp.ui.services import universe_service

logger = logging.getLogger(__name__)

_FREQ_MAP = {'Q': 'QE', 'A': 'YE', 'M': 'ME', 'W': 'W-FRI', 'D': 'B'}
_DENSE_FREQS = frozenset({'M', 'W', 'D'})  # carry-forward path (price sequences)
_EXPORT_DIR = Path(config.data.root_dir) / 'feature_exports'


# Raw price-panel columns added to every build alongside the registry factors.
PRICE_COLS = ['close', 'volume']

# Dense (daily) build column classes. TA comes dense from ta_panel; price-dependent
# factors are EXCLUDED from the dense palette (carrying a quarter-old pe/mktcap next
# to a daily close is incoherent — build price ratios from dense close instead).
_DENSE_TA_COLS = ['rsi_14', 'macd_hist', 'macd_norm', 'bb_pct', 'ma7_ma28', 'ma14_ma56']
_PRICE_DEP_COLS = ['mktcap', 'pe', 'pb', 'ps', 'ev_ebitda', 'ev_ebit', 'ev_sales',
                   'fcf_yield', 'mom_12_1', 'mom_6_1', 'vol_21d', 'ma200_ratio']


def available_columns(freq: str = 'A') -> list[str]:
    """Base feature palette for a grid frequency.

    Coarse (A/Q) = all registry factors + close/volume. Dense (D/W/M) =
    fundamentals-only carried factors + dense close/volume/TA (price-dependent
    factors dropped; users derive price ratios from dense close).
    """
    factors = [f.name for f in all_factors()]
    if freq in _DENSE_FREQS:
        keep = [c for c in factors if c not in _PRICE_DEP_COLS]
        return keep + PRICE_COLS
    return factors + PRICE_COLS


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


def _apply_steps(panel: pd.DataFrame, steps: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Apply steps, pruning output to produced columns.

    Steps whose input columns aren't in the panel (e.g. a price-ratio factor
    selected then the grid switched to a dense frequency) are SKIPPED with a
    warning instead of crashing the build.
    """
    sector_series = None
    feature_cols: list[str] = []
    skipped: list[str] = []
    for step in steps:
        missing = [c for c in _eng.step_input_cols(step) if c not in panel.columns]
        if missing:
            skipped.append(f'skipped {step.get("op")} on {",".join(missing)} '
                           f'(not available for this frequency)')
            continue
        if step.get('op') == 'norm' and step.get('method') == 'sector':
            if sector_series is None:
                from irp.query.simfin import sector_map
                sector_series = sector_map()
            step = {**step, 'sector': sector_series}
        panel = _eng.apply_step(panel, step)
        for c in _eng.step_output_cols(step):
            if c not in feature_cols:
                feature_cols.append(c)
    keep = ['Date', 'Ticker'] + [c for c in feature_cols if c in panel.columns]
    return panel[keep], skipped


_RESERVED_COLS = frozenset({'Date', 'Ticker', 'fwd_ret', 'label', 'split'})


def _split_and_scale(
    panel: pd.DataFrame, scale_cfg: dict | None, split_cfg: dict | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Assign a train/valid/test split and scale feature columns in place.

    Applied last (after steps + label). The split is computed first so the scaler
    (global/ticker scope) can fit on the TRAIN rows only — the leak-free protocol.
    Per-date scaling is unaffected by the split. A `split` column is appended when
    a split is configured; scaling never touches reserved columns.
    """
    notes: list[str] = []
    if panel.empty:
        return panel, notes

    split_method = (split_cfg or {}).get('method', 'none')
    split_series = None
    if split_method and split_method != 'none':
        ratios = (float(split_cfg.get('train', 0.7)), float(split_cfg.get('valid', 0.15)),
                  float(split_cfg.get('test', 0.15)))
        split_series = _eng.assign_split(panel, method=split_method, ratios=ratios,
                                         seed=int(split_cfg.get('seed', 0)))

    method = (scale_cfg or {}).get('method')
    if method and method != 'none':
        feat_cols = [c for c in panel.columns if c not in _RESERVED_COLS]
        if feat_cols:
            scope = scale_cfg.get('scope', 'date')
            cutoff = scale_cfg.get('train_cutoff')
            train_mask = None
            if scope in ('global', 'ticker') and split_series is not None:
                train_mask = (split_series == 'train')   # fit on the train split
                cutoff = None
            elif scope in ('global', 'ticker') and cutoff is None:
                notes.append(f'scaling ({scope}/{method}) fit on full sample — set a '
                             f'train cutoff year or a split to avoid look-ahead')
            panel = _eng.scale_features(panel, feat_cols, method=method, scope=scope,
                                        train_cutoff=cutoff, train_mask=train_mask)

    if split_series is not None:
        panel = panel.copy()
        panel['split'] = split_series.to_numpy()
    return panel, notes


def _load_snapshots_long(start_year, end_year, variant, cols) -> pd.DataFrame:
    """Stack all cached snapshots in [start, end] into a long (Date, Ticker, cols) frame.

    Lower bound reaches 2 years before `start_year` so early grid dates inherit the
    most recent prior filing via carry-forward (annual filings are ~1 year apart).
    """
    lo, hi = datetime.date(start_year - 2, 1, 1), datetime.date(end_year, 12, 31)
    root = _snapshot.CACHE_ROOT / variant
    parts = []
    if root.exists():
        for f in sorted(root.glob('*.parquet')):
            try:
                dt = datetime.date.fromisoformat(f.stem)
            except ValueError:
                continue
            if not (lo <= dt <= hi):
                continue
            s = _snapshot.load(dt, variant)
            if s is None or s.empty:
                continue
            s = s[[c for c in cols if c in s.columns]].reset_index()
            s.insert(0, 'Date', pd.Timestamp(dt))
            parts.append(s)
    if not parts:
        return pd.DataFrame(columns=['Date', 'Ticker'] + list(cols))
    return pd.concat(parts, ignore_index=True)


def _dense_ta(grid: list[datetime.date], tickers) -> pd.DataFrame:
    """PIT (backward as-of) TA snapshot from ta_panel at each grid date, per ticker."""
    import polars as pl

    from irp.panel.load import load_ta_panel
    ta = load_ta_panel()
    if ta.is_empty():
        return pd.DataFrame(columns=['Date', 'Ticker'] + _DENSE_TA_COLS)
    hi = pl.lit(pd.Timestamp(max(grid))).cast(pl.Datetime('ns'))
    ta = (ta.with_columns(pl.col('Date').cast(pl.Datetime('ns')))
            .filter(pl.col('Date') <= hi))
    if tickers is not None:
        ta = ta.filter(pl.col('Ticker').is_in(list(tickers)))
    grid_df = (pl.DataFrame({'Date': [pd.Timestamp(d) for d in grid]})
               .with_columns(pl.col('Date').cast(pl.Datetime('ns'))))
    tick = ta.select('Ticker').unique()
    spine = grid_df.join(tick, how='cross').sort('Date')
    joined = spine.join_asof(ta.sort('Date'), on='Date', by='Ticker', strategy='backward')
    out = joined.to_pandas()
    out['Date'] = pd.to_datetime(out['Date'])
    return out


def _build_dense(start_year, end_year, freq, variant, steps, label_cfg,
                 tickers, scale_cfg=None, split_cfg=None) -> tuple[pd.DataFrame, list[str]]:
    """Dense price-sequence build: dense close/volume/TA + carry-forward fundamentals."""
    grid = _date_grid(start_year, end_year, freq)
    if not grid:
        return pd.DataFrame(), []

    pv = _price_volume(grid, tickers=tickers)
    if pv.empty:
        return pd.DataFrame(), []
    pv['Date'] = pd.to_datetime(pv['Date'])
    spine = pv.sort_values(['Ticker', 'Date'])

    warnings: list[str] = []
    # carried = fundamentals only; TA comes dense from ta_panel (avoid column collision)
    carried = [c for c in available_columns(freq)
               if c not in PRICE_COLS and c not in _DENSE_TA_COLS]
    snap_long = _load_snapshots_long(start_year, end_year, variant, carried)
    if snap_long.empty:
        warnings.append(f'no {variant} fundamental snapshots in range (fundamentals NaN)')
    else:
        spine = _eng.asof_join(spine, snap_long, by='Ticker')

    ta = _dense_ta(grid, tickers)
    if not ta.empty:
        spine = _eng.asof_join(spine, ta, by='Ticker')

    panel = spine.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    panel['Date'] = panel['Date'].dt.date

    panel, skipped = _apply_steps(panel, steps)
    warnings += skipped

    mode = label_cfg.get('mode', 'none')
    if mode != 'none':
        fwd = _forward_returns(grid, int(label_cfg.get('horizon_days', 252)), tickers=tickers)
        panel = _eng.attach_label(panel, fwd, mode=mode,
                                  n_buckets=int(label_cfg.get('n_buckets', 5)))
    panel, snotes = _split_and_scale(panel, scale_cfg, split_cfg)
    return panel, warnings + snotes


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
    scale_cfg: dict | None = None,
    split_cfg: dict | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the long (Date, Ticker) feature panel.

    Two paths by `freq`:
      - Coarse (A/Q): one row per ticker at each snapshot rebalance date (the
        cross-sectional panel). Cache-first; cold range returns `missing_dates`.
      - Dense (D/W/M): a price *sequence* per ticker — dense close/volume/TA from
        the daily panels + carry-forward (PIT) fundamentals from snapshots.
    Returns (panel, notes) where `notes` is missing dates (coarse) or warnings (dense).
    """
    tickers = universe_service._filter_tickers(market, sector, watchlist)

    if freq in _DENSE_FREQS:
        return _build_dense(start_year, end_year, freq, variant, steps, label_cfg,
                            tickers, scale_cfg, split_cfg)

    grid = _date_grid(start_year, end_year, freq)
    if not grid:
        return pd.DataFrame(), []

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

    # Add raw close + volume (PIT per grid date) as base columns, so price/volume
    # temporal & math ops (returns, volume change, dollar volume) are available.
    built_dates = sorted(snapshots.keys())
    pv = _price_volume(built_dates, tickers=tickers)
    panel = panel.merge(pv, on=['Date', 'Ticker'], how='left')

    # Apply steps; output pruned to produced columns (the 39 base columns are
    # inputs, not output). Steps with absent inputs are skipped with a warning.
    panel, skipped = _apply_steps(panel, steps)
    missing += skipped

    mode = label_cfg.get('mode', 'none')
    if mode != 'none':
        horizon = int(label_cfg.get('horizon_days', 252))
        fwd = _forward_returns(grid, horizon, tickers=tickers)
        panel = _eng.attach_label(
            panel, fwd, mode=mode, n_buckets=int(label_cfg.get('n_buckets', 5))
        )
    panel, snotes = _split_and_scale(panel, scale_cfg, split_cfg)
    return panel, missing + snotes


def _write_frame(df: pd.DataFrame, fmt: str, stem: str) -> Path:
    path = _EXPORT_DIR / f'{stem}.{fmt}'
    if fmt == 'csv':
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path


def export_panel(
    df: pd.DataFrame, fmt: Literal['parquet', 'csv'], name: str = 'features',
) -> Path | list[Path]:
    """Write the panel to data/feature_exports/.

    No `split` column → one file `<name>_<ts>.<fmt>` (returns a Path). With a
    `split` column → one file per split `<name>_<ts>_<split>.<fmt>` (train/valid/
    test, in that order; the `split` column is dropped from each); returns a list.
    """
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    safe = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in name) or 'features'
    if 'split' not in df.columns:
        return _write_frame(df, fmt, f'{safe}_{ts}')
    paths = []
    for split_name in ('train', 'valid', 'test'):
        part = df[df['split'] == split_name].drop(columns=['split'])
        if part.empty:
            continue
        paths.append(_write_frame(part, fmt, f'{safe}_{ts}_{split_name}'))
    return paths


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
