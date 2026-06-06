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

import numpy as np
import pandas as pd

from irp.features import normalize as _norm

_ROLL_FNS = ('mean', 'std', 'min', 'max', 'sum')


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

    if scope == 'date':
        for _, idx in out.groupby('Date', sort=False).groups.items():
            for c in cols:
                s = out.loc[idx, c]
                out.loc[idx, c] = _apply_scale(s, *_scale_params(s, method))
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

def step_output_cols(step: dict) -> list[str]:
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


def step_input_cols(step: dict) -> list[str]:
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


def apply_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
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
