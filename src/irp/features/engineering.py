"""Feature-engineering ops on a long (Date, Ticker) panel.

Pure functions; no DB, no Dash. Each op takes a long-format DataFrame (one
row per Date×Ticker, with feature columns) and returns a new DataFrame with
the derived column(s) added. The ordered list of `step` dicts is the recipe;
`apply_step` dispatches one step.

PIT / staleness caveat
----------------------
Fundamentals enter the panel via grid-sampled PIT snapshots: at two
consecutive grid dates the PIT lookup can return the *same* filing (no new
publish in between). Temporal ops (lag/diff/pct_change/rolling) on fundamental
columns therefore carry sampling staleness — e.g. a year-over-year diff can be
a spurious 0 when no new filing landed. Price/volume/TA columns are dense and
clean. Users wanting leak-free fundamental growth should prefer the PIT-safe
registry factors already in the base palette (`rev_growth_1y`,
`earn_growth_1y`, `mom_*`).
"""
import datetime
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd

from irp.features import normalize as _norm

_ROLL_FNS = ('mean', 'std', 'min', 'max', 'sum')


class StepDict(TypedDict, total=False):
    """One recipe step. `op` selects the operation; the other keys are the
    operands that op reads (see `apply_step`). All optional — the relevant
    subset depends on `op`. This is the on-the-wire shape of a recipe step.
    """
    op: str                      # base|lag|diff|pctchange|lagwin|rolling|ratio|product|linear|log|winsorize|norm
    col: str                     # single-column ops
    cols: list[str]              # base / norm (multi-column)
    a: str                       # ratio / product — left/numerator
    b: str                       # ratio / product — right/denominator
    k: int                       # lag / diff / pctchange period
    n: int                       # lag-window length
    window: int                  # rolling window
    fn: str                      # rolling reducer (mean|std|min|max|sum)
    weights: dict[str, float]    # linear combination weights
    name: str                    # linear output column name
    p: float                     # winsorize tail fraction
    method: str                  # norm method (zscore|rank|sector)
    sector: object               # norm sector Series (injected by the service)


# ── panel assembly ────────────────────────────────────────────────────

def assemble_panel(
    snapshots: dict[datetime.date, pd.DataFrame],
) -> pd.DataFrame:
    """Stack per-date cross-sections (indexed by Ticker) into a long panel.

    Output columns: Date, Ticker, <all snapshot columns>. Sorted by
    (Ticker, Date) so per-ticker temporal ops are well ordered.
    """
    if not snapshots:
        return pd.DataFrame(columns=['Date', 'Ticker'])
    parts = []
    for d, snap in snapshots.items():
        part = snap.reset_index()
        part.insert(0, 'Date', d)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(['Ticker', 'Date']).reset_index(drop=True)


# ── temporal ops (per-ticker, ordered by Date) ────────────────────────

def _grouped(df: pd.DataFrame, col: str):
    return df.groupby('Ticker', sort=False)[col]


def add_lag(df: pd.DataFrame, col: str, k: int) -> pd.DataFrame:
    out = df.copy()
    out[f'{col}_lag{k}'] = _grouped(out, col).shift(k)
    return out


def add_diff(df: pd.DataFrame, col: str, k: int) -> pd.DataFrame:
    out = df.copy()
    out[f'{col}_diff{k}'] = _grouped(out, col).diff(k)
    return out


def add_pct_change(df: pd.DataFrame, col: str, k: int) -> pd.DataFrame:
    out = df.copy()
    out[f'{col}_pct{k}'] = _grouped(out, col).pct_change(k).replace(
        [np.inf, -np.inf], np.nan
    )
    return out


def add_lag_window(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    """Level + n lags as columns: col, col_lag1 … col_lag_n (a price window per row)."""
    out = df.copy()
    for k in range(1, int(n) + 1):
        out[f'{col}_lag{k}'] = _grouped(out, col).shift(k)
    return out


def add_rolling(df: pd.DataFrame, col: str, window: int, fn: str) -> pd.DataFrame:
    if fn not in _ROLL_FNS:
        raise ValueError(f'unknown rolling fn {fn!r}; expected one of {_ROLL_FNS}')
    out = df.copy()
    rolled = _grouped(out, col).rolling(window, min_periods=window)
    out[f'{col}_roll{window}{fn}'] = getattr(rolled, fn)().reset_index(level=0, drop=True)
    return out


# ── math / interaction ops (row-wise) ─────────────────────────────────

def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    with np.errstate(divide='ignore', invalid='ignore'):
        out = num / denom
    return out.replace([np.inf, -np.inf], np.nan)


def add_ratio(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    out = df.copy()
    out[f'{a}_over_{b}'] = _safe_div(out[a], out[b])
    return out


def add_product(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    out = df.copy()
    out[f'{a}_x_{b}'] = out[a] * out[b]
    return out


def add_linear(
    df: pd.DataFrame, weights: dict[str, float], name: str = 'linear'
) -> pd.DataFrame:
    missing = [c for c in weights if c not in df.columns]
    if missing:
        raise ValueError(f'linear: missing columns {missing}')
    out = df.copy()
    acc = pd.Series(0.0, index=out.index)
    for col, w in weights.items():
        acc = acc + out[col].astype('float64') * float(w)
    out[name] = acc
    return out


def add_log(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    s = out[col].astype('float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        out[f'log_{col}'] = np.where(s > 0, np.log(s), np.nan)
    return out


def add_winsorize(df: pd.DataFrame, col: str, p: float) -> pd.DataFrame:
    """Clip column to its [p, 1-p] quantiles (computed over the whole panel)."""
    out = df.copy()
    lo, hi = out[col].quantile(p), out[col].quantile(1 - p)
    out[f'{col}_wins'] = out[col].clip(lo, hi)
    return out


def signed_log(s: pd.Series) -> pd.Series:
    """sign(x) · log1p(|x|). Monotonic, stateless, maps 0 -> 0; tames heavy tails
    without needing fit params, so it is train/test consistent by construction."""
    s = s.astype('float64')
    return np.sign(s) * np.log1p(np.abs(s))


def detect_heavy_tailed(
    df: pd.DataFrame, cols: list[str], thresh: float = 20.0
) -> list[str]:
    """Columns whose tail-ratio (q99.9 - q0.1) / IQR exceeds `thresh`.

    Robust/minmax scaling does not clip, so heavy-tailed columns stay huge after
    scaling. This flags them (drives the pre-scale warning + the user's column
    pick). Constant columns (zero IQR) are never flagged.
    """
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].astype('float64')
        iqr = s.quantile(0.75) - s.quantile(0.25)
        if iqr == 0 or pd.isna(iqr):
            continue
        tail = s.quantile(0.999) - s.quantile(0.001)
        if tail / iqr > thresh:
            out.append(c)
    return out


_TAME_RESERVED = frozenset({'Date', 'Ticker', 'fwd_ret', 'label', 'split'})


def tame_columns(
    df: pd.DataFrame,
    cols: list[str],
    action: str,
    p: float = 0.01,
    train_mask: pd.Series | None = None,
    side: str = 'both',
) -> pd.DataFrame:
    """Apply one heavy-tail action to `cols` before scaling. Reserved columns are
    never touched even if passed.

    action:
        'clip' — winsorize to the [p, 1-p] quantiles computed on `train_mask` rows
                 (full sample if no mask), applied to all rows. Leak-free w.r.t.
                 the test split. `side` ∈ {'both','lower','upper'} clips only that
                 tail — use a one-sided clip when only one tail is bad so the clean
                 tail's data is left untouched.
        'log'  — signed_log in place (stateless).
        'drop' — remove the columns.
        'none' — no-op.
    """
    cols = [c for c in cols if c in df.columns and c not in _TAME_RESERVED]
    if not cols or action in (None, 'none'):
        return df
    out = df.copy()
    if action == 'drop':
        return out.drop(columns=cols)
    if action == 'log':
        for c in cols:
            out[c] = signed_log(out[c])
        return out
    if action == 'clip':
        p = min(max(float(p), 0.0), 0.4999)   # clamp to a valid tail fraction
        out[cols] = out[cols].astype('float64')
        fit = out.loc[train_mask.reindex(out.index).fillna(False).astype(bool)] if train_mask is not None else out
        for c in cols:
            src = fit[c] if not fit[c].dropna().empty else out[c]
            lo = src.quantile(p) if side in ('both', 'lower') else None
            hi = src.quantile(1 - p) if side in ('both', 'upper') else None
            out[c] = out[c].clip(lower=lo, upper=hi)
        return out
    raise ValueError(f'unknown tame action {action!r}')


def residual_scale_flags(
    df: pd.DataFrame, cols: list[str], thresh: float = 10.0
) -> dict[str, float]:
    """Post-scale check: {col: p99_abs} for feature columns whose 99th-percentile
    absolute value still exceeds `thresh` (scaling failed to tame them)."""
    flags = {}
    for c in cols:
        if c not in df.columns or c in _TAME_RESERVED:
            continue
        p99 = df[c].astype('float64').abs().quantile(0.99)
        if pd.notna(p99) and p99 > thresh:
            flags[c] = float(p99)
    return flags


# ── heavy-tail inspection + per-column tame plan ──────────────────────

_OUTLIER_Z = 3.5  # robust-z magnitude above which a row counts as an outlier


@dataclass
class HeavyTailReport:
    """Three views of the heavy-tailed columns, so a human can choose drop/log/clip.

    summary  — one row per flagged column: col, n, median, p99, max, iqr, tail_ratio,
               n_outliers, worst_ticker (ticker owning the single most-extreme row).
    offenders — long: col, Date, Ticker, value, z (robust z = (x-median)/IQR), the
               top_n rows per column by |z|.
    by_ticker — col, Ticker, n_outliers, max_abs_z: spot a single bad ticker.
    """
    summary: pd.DataFrame
    offenders: pd.DataFrame
    by_ticker: pd.DataFrame


def heavy_tail_report(
    df: pd.DataFrame, cols: list[str] | None = None,
    thresh: float = 20.0, top_n: int = 20, with_detail: bool = True,
) -> HeavyTailReport:
    """Inspect heavy-tailed columns. When `cols` is None, auto-picks the flagged
    columns via `detect_heavy_tailed` (same tail-ratio metric). `with_detail=False`
    returns only the summary (skips the per-column top_n sort + by-ticker group —
    use it for the overview pass on huge panels; drill into one column for detail)."""
    if cols is None:
        candidates = [c for c in df.columns if c not in _TAME_RESERVED]
        cols = detect_heavy_tailed(df, candidates, thresh)
    summ, off, byt = [], [], []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].astype('float64')
        med = s.median()
        iqr = s.quantile(0.75) - s.quantile(0.25)
        z = (s - med) / iqr if iqr else pd.Series(0.0, index=s.index)
        absz = z.abs()
        out_mask = absz > _OUTLIER_Z
        worst_ticker = df.loc[absz.idxmax(), 'Ticker'] if len(absz) and absz.notna().any() else None
        tail = s.quantile(0.999) - s.quantile(0.001)
        summ.append({
            'col': c, 'n': int(s.notna().sum()), 'median': float(med),
            'p1': float(s.quantile(0.01)), 'p99': float(s.quantile(0.99)),
            'min': float(s.min()), 'max': float(s.max()),
            'iqr': float(iqr), 'tail_ratio': float(tail / iqr) if iqr else np.nan,
            'n_outliers': int(out_mask.sum()), 'worst_ticker': worst_ticker,
        })
        if not with_detail:
            continue
        sub = pd.DataFrame({'col': c, 'Date': df['Date'], 'Ticker': df['Ticker'],
                            'value': s, 'z': z})
        off.append(sub.reindex(absz.sort_values(ascending=False).index).head(top_n))
        if out_mask.any():
            tick = df['Ticker']
            g = pd.DataFrame({'Ticker': tick[out_mask], 'absz': absz[out_mask], 'value': s[out_mask]})
            # worst row per ticker = the actual value at its largest |z| (signed, real)
            worst_val = g.loc[g.groupby('Ticker')['absz'].idxmax()].set_index('Ticker')['value']
            totals = tick[s.notna()].value_counts()         # ticker's non-null rows in this col
            agg = g.groupby('Ticker', sort=False).size().rename('n_outliers').reset_index()
            agg['n_rows'] = agg['Ticker'].map(totals).astype(int)
            agg['pct_outliers'] = (agg['n_outliers'] / agg['n_rows']).round(4)
            agg['worst_value'] = agg['Ticker'].map(worst_val)
            agg.insert(0, 'col', c)
            byt.append(agg.sort_values('pct_outliers', ascending=False)[
                ['col', 'Ticker', 'n_rows', 'n_outliers', 'pct_outliers', 'worst_value']])
    summary = (pd.DataFrame(summ) if summ else
               pd.DataFrame(columns=['col', 'n', 'median', 'p1', 'p99', 'min', 'max', 'iqr',
                                     'tail_ratio', 'n_outliers', 'worst_ticker']))
    offenders = (pd.concat(off, ignore_index=True) if off else
                 pd.DataFrame(columns=['col', 'Date', 'Ticker', 'value', 'z']))
    by_ticker = (pd.concat(byt, ignore_index=True) if byt else
                 pd.DataFrame(columns=['col', 'Ticker', 'n_rows', 'n_outliers',
                                       'pct_outliers', 'worst_value']))
    return HeavyTailReport(summary, offenders, by_ticker)


def apply_tame_plan(
    df: pd.DataFrame, plan: list[dict], train_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply a per-column tame plan before scaling. `plan` = list of
    `{col, action, p}` with action in {drop, log, clip}; different columns may take
    different actions in one pass. Delegates to `tame_columns` (clip fits on
    `train_mask` rows, per-column `p`). Reserved columns are never touched."""
    out = df
    drops = [s['col'] for s in plan if s.get('action') == 'drop']
    logs = [s['col'] for s in plan if s.get('action') == 'log']
    if drops:
        out = tame_columns(out, drops, 'drop')
    if logs:
        out = tame_columns(out, logs, 'log')
    for s in plan:
        if s.get('action') == 'clip':
            out = tame_columns(out, [s['col']], 'clip', p=float(s.get('p', 0.01)),
                               train_mask=train_mask, side=s.get('side', 'both'))
    return out


# ── feature scaling (whole-matrix, model preprocessing) ───────────────

_SCALE_METHODS = ('minmax', 'robust')


def _scale_params(s: pd.Series, method: str) -> tuple[float, float]:
    """(center, scale) for a fit series. NaN-skipping; scale may be 0 (constant)."""
    if method == 'minmax':
        return s.min(), s.max() - s.min()
    if method == 'robust':
        return s.median(), s.quantile(0.75) - s.quantile(0.25)
    raise ValueError(f'unknown scale method {method!r}; expected one of {_SCALE_METHODS}')


def _apply_scale(s: pd.Series, center: float, scale: float) -> pd.Series:
    """(s - center) / scale. Zero/NaN scale (constant group) → 0 where present, NaN where absent."""
    s = s.astype('float64')
    if scale == 0 or pd.isna(scale):
        return pd.Series(np.where(s.notna(), 0.0, np.nan), index=s.index)
    return _safe_div(s - center, scale)


def scale_features(
    df: pd.DataFrame,
    cols: list[str],
    method: str = 'minmax',
    scope: str = 'date',
    train_cutoff: int | None = None,
    train_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Scale feature columns IN PLACE (model preprocessing), grouped by `scope`.

    method: 'minmax' → [0, 1]; 'robust' → (x − median) / IQR.
    scope:
        'date'   — fit within each Date cross-section. Unconditionally leak-free
                   for walk-forward (training on Date<d never sees d's scale).
        'global' — fit one scaler on the TRAIN rows (Date.year ≤ train_cutoff),
                   apply to all rows. Leak-free only w.r.t. the post-cutoff test
                   split — a fixed single train/test split, NOT expanding
                   walk-forward (pre-cutoff rows still see up-to-cutoff stats).
        'ticker' — like 'global' but fit per ticker on its own train rows.
    train_cutoff: calendar year; train = rows with Date.year ≤ cutoff. Ignored
        for 'date'. None for global/ticker → fit on full sample (look-ahead).
    train_mask: explicit boolean Series (aligned to df.index) marking the train
        rows — e.g. the train split. Overrides `train_cutoff` when given. This is
        how the service fits the scaler on the train split.
    Constant group (range/IQR 0) → 0. Input NaN preserved. Reserved columns
    (Date/Ticker/fwd_ret/label) are scaled only if the caller passes them in
    `cols` (the service excludes them).
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    # Scaled values are continuous float64; upcast the targets first so partial-row
    # .loc assignment never hits pandas' float64→float32 LossySetitemError (dense
    # panels carry float32 columns).
    out[cols] = out[cols].astype('float64')

    if scope == 'date':
        # Vectorized per-date fit: one groupby-transform per stat for the whole block,
        # instead of a dates × cols Python loop with .loc assignment. Same semantics
        # as `_apply_scale`: constant group (range/IQR 0) → 0 where present, NaN where
        # absent; input NaN preserved; non-finite → NaN.
        g = out.groupby('Date', sort=False)[cols]
        if method == 'minmax':
            center = g.transform('min')
            scale = g.transform('max') - center
        elif method == 'robust':
            center = g.transform('median')
            scale = (g.transform(lambda s: s.quantile(0.75))
                     - g.transform(lambda s: s.quantile(0.25)))
        else:
            raise ValueError(f'unknown scale method {method!r}; expected one of {_SCALE_METHODS}')
        block = out[cols]
        res = (block - center) / scale
        res = res.where(~(scale.eq(0) | scale.isna()), 0.0)   # constant group → 0
        res = res.where(block.notna(), np.nan)                # preserve NaN input
        res = res.replace([np.inf, -np.inf], np.nan)
        out[cols] = res
        return out

    if train_mask is not None:
        mask = train_mask.reindex(out.index).fillna(False).astype(bool)
    elif train_cutoff is not None:
        mask = pd.to_datetime(out['Date']).dt.year <= int(train_cutoff)
    else:
        mask = pd.Series(True, index=out.index)

    if scope == 'global':
        for c in cols:
            train_s = out.loc[mask, c]
            if train_s.dropna().empty:
                train_s = out[c]
            out[c] = _apply_scale(out[c], *_scale_params(train_s, method))
        return out

    if scope == 'ticker':
        for _, idx in out.groupby('Ticker', sort=False).groups.items():
            tmask = mask.loc[idx].to_numpy()
            for c in cols:
                col_all = out.loc[idx, c]
                train_s = col_all[tmask]
                if train_s.dropna().empty:
                    train_s = col_all          # IPO ticker w/ no train rows
                out.loc[idx, c] = _apply_scale(col_all, *_scale_params(train_s, method))
        return out

    raise ValueError(f'unknown scale scope {scope!r}')


# ── train / valid / test split ────────────────────────────────────────

def assign_split(
    df: pd.DataFrame,
    method: str = 'date',
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> pd.Series:
    """Return a 'train'/'valid'/'test' label Series aligned to `df.index`.

    method='date'   — chronological by UNIQUE date: earliest ratios[0] of dates →
                      train, next ratios[1] → valid, rest → test (train < valid <
                      test, test most recent). Every row of a date shares a split.
    method='ticker' — leave-tickers-out: each ticker assigned wholly to one split
                      by a seeded shuffle, partitioned by `ratios`. No ticker spans
                      splits (group-disjoint).
    `ratios` need not sum to 1 (normalized). The valid boundary uses round();
    test takes the remainder so all three are non-empty when the unit count allows.
    """
    total = float(sum(ratios)) or 1.0
    r_train, r_valid = ratios[0] / total, ratios[1] / total

    if method == 'date':
        units = sorted(df['Date'].unique())
    elif method == 'ticker':
        units = sorted(df['Ticker'].unique())
        rng = np.random.default_rng(seed)
        units = list(np.array(units, dtype=object)[rng.permutation(len(units))])
    else:
        raise ValueError(f'unknown split method {method!r}')

    n = len(units)
    n_train = round(n * r_train)
    n_valid = round(n * r_valid)
    # keep all three non-empty when the unit count allows (n >= 3): a tiny or
    # skewed valid/test fraction can round to 0 — borrow from train.
    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)      # leave ≥1 each for valid + test
        n_valid = max(n_valid, 1)
        if n_train + n_valid > n - 1:              # ensure ≥1 left for test
            n_valid = n - 1 - n_train
    label = {}
    for i, u in enumerate(units):
        if i < n_train:
            label[u] = 'train'
        elif i < n_train + n_valid:
            label[u] = 'valid'
        else:
            label[u] = 'test'
    key = 'Date' if method == 'date' else 'Ticker'
    return df[key].map(label)


# ── cross-sectional normalization (per Date) ──────────────────────────

_NORM_SUFFIX = {'zscore': '_z', 'rank': '_rank', 'sector': '_sn'}


def normalize_step(
    df: pd.DataFrame,
    cols: list[str],
    method: str = 'zscore',
    sector: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply a cross-sectional normalization per Date, writing new columns.

    method: 'zscore' | 'rank' | 'sector' (sector requires a Ticker->sector map).
    """
    suffix = _NORM_SUFFIX.get(method)
    if suffix is None:
        raise ValueError(f'unknown norm method {method!r}')
    out = df.copy()
    new_cols = [f'{c}{suffix}' for c in cols]

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        g = group.set_index('Ticker')
        if method == 'zscore':
            res = _norm.zscore(g, cols)
        elif method == 'rank':
            res = _norm.rank_norm(g, cols)
        else:
            res = _norm.sector_neutral(g, sector, cols)
        return res[cols].reset_index(drop=True)

    for c in cols:
        out[f'{c}{suffix}'] = np.nan
    for _, idx in out.groupby('Date', sort=False).groups.items():
        block = out.loc[idx]
        normed = _apply(block)
        for src, dst in zip(cols, new_cols, strict=False):
            out.loc[idx, dst] = normed[src].to_numpy()
    return out


# ── step dispatch ─────────────────────────────────────────────────────

def step_output_cols(step: StepDict) -> list[str]:
    """Names of the feature column(s) a step contributes to the output.

    Source columns are NOT included — only what the step produces (or, for
    `base`, the raw column the user explicitly chose to keep).
    """
    op = step.get('op')
    if op == 'base':
        return [step['col']]
    if op in ('lag', 'diff', 'pct_change'):
        suffix = {'lag': 'lag', 'diff': 'diff', 'pct_change': 'pct'}[op]
        return [f'{step["col"]}_{suffix}{int(step.get("k", 1))}']
    if op == 'lagwin':
        c, n = step['col'], int(step.get('n', step.get('window', 1)))
        return [c] + [f'{c}_lag{k}' for k in range(1, n + 1)]
    if op == 'rolling':
        return [f'{step["col"]}_roll{int(step["window"])}{step.get("fn", "mean")}']
    if op == 'ratio':
        return [f'{step["a"]}_over_{step["b"]}']
    if op == 'product':
        return [f'{step["a"]}_x_{step["b"]}']
    if op == 'linear':
        return [step.get('name', 'linear')]
    if op == 'log':
        return [f'log_{step["col"]}']
    if op == 'winsorize':
        return [f'{step["col"]}_wins']
    if op == 'norm':
        suffix = _NORM_SUFFIX.get(step.get('method', 'zscore'), '_z')
        return [f'{c}{suffix}' for c in step['cols']]
    raise ValueError(f'unknown op {op!r}')


def step_input_cols(step: StepDict) -> list[str]:
    """Columns a step READS (its inputs) — used to skip steps whose inputs are absent."""
    op = step.get('op')
    if op in ('ratio', 'product'):
        return [step['a'], step['b']]
    if op == 'norm':
        return list(step.get('cols', []))
    if op in ('lag', 'diff', 'pct_change', 'rolling', 'log', 'winsorize', 'lagwin', 'base'):
        return [step['col']]
    if op == 'linear':
        return list(step.get('weights', {}).keys())
    return []


def apply_step(df: pd.DataFrame, step: StepDict) -> pd.DataFrame:
    """Apply one recipe step. `step['op']` selects the operation."""
    op = step.get('op')
    if op == 'base':
        col = step['col']
        if col not in df.columns:
            raise ValueError(f'base column {col!r} not in panel')
        return df  # base columns are already present; this is a declared include
    if op == 'lag':
        return add_lag(df, step['col'], int(step.get('k', 1)))
    if op == 'diff':
        return add_diff(df, step['col'], int(step.get('k', 1)))
    if op == 'pct_change':
        return add_pct_change(df, step['col'], int(step.get('k', 1)))
    if op == 'lagwin':
        return add_lag_window(df, step['col'], int(step.get('n', step.get('window', 1))))
    if op == 'rolling':
        return add_rolling(df, step['col'], int(step['window']), step.get('fn', 'mean'))
    if op == 'ratio':
        return add_ratio(df, step['a'], step['b'])
    if op == 'product':
        return add_product(df, step['a'], step['b'])
    if op == 'linear':
        return add_linear(df, step['weights'], step.get('name', 'linear'))
    if op == 'log':
        return add_log(df, step['col'])
    if op == 'winsorize':
        return add_winsorize(df, step['col'], float(step.get('p', 0.01)))
    if op == 'norm':
        return normalize_step(
            df, step['cols'], step.get('method', 'zscore'), step.get('sector')
        )
    raise ValueError(f'unknown op {op!r}')


# ── PIT carry-forward (for dense sequence builds) ─────────────────────

def asof_join(left: pd.DataFrame, right: pd.DataFrame, by: str = 'Ticker') -> pd.DataFrame:
    """Backward as-of merge: each left row gets the last right row with Date <= it.

    Per `by` group, no look-ahead. Both frames are sorted on Date internally.
    Used to carry forward filing/snapshot values onto a dense price grid.
    """
    if right.empty:
        return left.copy()
    left = left.copy()
    right = right.copy()
    left['Date'] = left['Date'].astype('datetime64[ns]')
    right['Date'] = right['Date'].astype('datetime64[ns]')
    left = left.sort_values('Date')
    right = right.sort_values('Date')
    return pd.merge_asof(left, right, on='Date', by=by, direction='backward')


# ── label attachment (per-date bucketing, no look-ahead) ──────────────

def attach_label(
    df: pd.DataFrame,
    fwd: pd.DataFrame,
    mode: str = 'continuous',
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Join the forward-return label.

    mode:
        'none'       — no label column added.
        'continuous' — raw `fwd_ret`.
        'binary'     — 1 if fwd_ret > per-date median else 0.
        'quantile'   — per-date qcut bucket in [0, n_buckets-1].

    Classification targets are bucketed *within each Date* to avoid using the
    full-sample (look-ahead) distribution.
    """
    if mode == 'none':
        return df
    out = df.merge(fwd, on=['Date', 'Ticker'], how='left')
    if mode == 'continuous':
        return out
    if mode == 'binary':
        med = out.groupby('Date')['fwd_ret'].transform('median')
        out['label'] = (out['fwd_ret'] > med).astype('int64')
        out.loc[out['fwd_ret'].isna(), 'label'] = np.nan
        return out
    if mode == 'quantile':
        def _bucket(s: pd.Series) -> pd.Series:
            try:
                return pd.qcut(s, n_buckets, labels=False, duplicates='drop')
            except ValueError:
                return pd.Series(np.nan, index=s.index)
        out['label'] = out.groupby('Date')['fwd_ret'].transform(_bucket)
        return out
    raise ValueError(f'unknown label mode {mode!r}')
