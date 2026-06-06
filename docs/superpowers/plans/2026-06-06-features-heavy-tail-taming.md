# Heavy-Tail Taming on /features — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `/features` detect heavy-tailed feature columns and clip / log / drop them before scaling, fit on the train split, with non-blocking before/after warnings — all under explicit user control.

**Architecture:** Three pure functions in `engineering.py` (detect, tame, residual-check) wrapped into the existing `_split_and_scale` stage of `features_service.py`; new Dash controls in the "3b · Scale features" block thread a `tame_*` config through the `scale` dict into the service. No notebook or library-default scaling — scaling stays owned by the page.

**Tech Stack:** Python 3.13, pandas, numpy, Dash 4, pytest, ruff, mypy, `uv`.

**Commit note:** This repo requires explicit user permission before each commit. Treat every "Commit" step as "ask the user, then commit on approval".

---

### Task 1: `signed_log` + `detect_heavy_tailed`

**Files:**
- Modify: `src/irp/features/engineering.py` (add after `add_winsorize`, ~line 162)
- Test: `tests/test_features_scaling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features_scaling.py`:

```python
def test_signed_log_monotonic_and_handles_neg_zero():
    s = pd.Series([-100.0, -1.0, 0.0, 1.0, 100.0])
    out = eng.signed_log(s)
    assert out.iloc[2] == 0.0                      # log1p(0) == 0
    assert (out.iloc[0] < 0) and (out.iloc[4] > 0)  # sign preserved
    assert out.is_monotonic_increasing              # monotonic


def test_detect_heavy_tailed_flags_fat_ignores_bounded_and_constant():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({
        'Date': [datetime.date(2020, 1, 1)] * n,
        'Ticker': [f'T{i}' for i in range(n)],
        'fat': rng.standard_cauchy(n) * 1e6,        # heavy tails
        'bounded': rng.uniform(-1, 1, n),           # well-behaved
        'const': np.ones(n),                        # zero IQR
    })
    flagged = eng.detect_heavy_tailed(df, ['fat', 'bounded', 'const'])
    assert 'fat' in flagged
    assert 'bounded' not in flagged
    assert 'const' not in flagged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features_scaling.py -k "signed_log or heavy_tailed" -v`
Expected: FAIL — `module 'irp.features.engineering' has no attribute 'signed_log'`.

- [ ] **Step 3: Write minimal implementation**

In `src/irp/features/engineering.py`, after `add_winsorize` (line 162):

```python
def signed_log(s: pd.Series) -> pd.Series:
    """sign(x) · log1p(|x|). Monotonic, stateless, maps 0 → 0; tames heavy tails
    without needing fit params, so it is train/test consistent by construction."""
    s = s.astype('float64')
    return np.sign(s) * np.log1p(np.abs(s))


def detect_heavy_tailed(
    df: pd.DataFrame, cols: list[str], thresh: float = 20.0
) -> list[str]:
    """Columns whose tail-ratio (q99.9 − q0.1) / IQR exceeds `thresh`.

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features_scaling.py -k "signed_log or heavy_tailed" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/irp/features/engineering.py tests/test_features_scaling.py
git commit -m "feat: add signed_log and detect_heavy_tailed feature helpers"
```

---

### Task 2: `tame_columns` (clip / log / drop, train-fit)

**Files:**
- Modify: `src/irp/features/engineering.py` (add after `detect_heavy_tailed`)
- Test: `tests/test_features_scaling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features_scaling.py`:

```python
def _tame_panel():
    """2 dates × 3 tickers; 'f' has a big outlier on the later (test) date."""
    rows = []
    for y, vals in ((2018, [1.0, 2.0, 3.0]), (2020, [4.0, 5.0, 1000.0])):
        d = datetime.date(y, 12, 31)
        for tk, v in zip(['A', 'B', 'C'], vals):
            rows.append({'Date': d, 'Ticker': tk, 'f': v, 'fwd_ret': 0.0, 'label': 0})
    return pd.DataFrame(rows)


def test_tame_clip_fits_on_train_no_leak():
    df = _tame_panel()
    train_mask = pd.to_datetime(df['Date']).dt.year <= 2018   # 2018 rows only
    out = eng.tame_columns(df, ['f'], 'clip', p=0.0, train_mask=train_mask)
    # train (2018) max is 3.0 → cap is 3.0; the 1000.0 test outlier clips to 3.0
    assert out['f'].max() == 3.0
    # perturbing the test outlier higher must not move the (train-fit) cap
    df2 = df.copy()
    df2.loc[df2['f'] == 1000.0, 'f'] = 1e9
    out2 = eng.tame_columns(df2, ['f'], 'clip', p=0.0, train_mask=train_mask)
    assert out2['f'].max() == 3.0


def test_tame_log_applies_signed_log():
    df = _tame_panel()
    out = eng.tame_columns(df, ['f'], 'log')
    assert np.isclose(out.loc[out['f'].idxmax(), 'f'], np.log1p(1000.0))


def test_tame_drop_removes_col_never_reserved():
    df = _tame_panel()
    out = eng.tame_columns(df, ['f', 'label'], 'drop')   # 'label' is reserved
    assert 'f' not in out.columns
    assert 'label' in out.columns                        # reserved never dropped


def test_tame_none_or_empty_is_noop():
    df = _tame_panel()
    assert eng.tame_columns(df, [], 'clip').equals(df)
    assert eng.tame_columns(df, ['f'], 'none').equals(df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features_scaling.py -k tame -v`
Expected: FAIL — `has no attribute 'tame_columns'`.

- [ ] **Step 3: Write minimal implementation**

In `src/irp/features/engineering.py`, after `detect_heavy_tailed`:

```python
_TAME_RESERVED = frozenset({'Date', 'Ticker', 'fwd_ret', 'label', 'split'})


def tame_columns(
    df: pd.DataFrame,
    cols: list[str],
    action: str,
    p: float = 0.01,
    train_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply one heavy-tail action to `cols` before scaling. Reserved columns are
    never touched even if passed.

    action:
        'clip' — winsorize to [p, 1−p] quantiles computed on `train_mask` rows
                 (full sample if no mask), applied to all rows. Leak-free w.r.t.
                 the test split.
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
        out[cols] = out[cols].astype('float64')
        if train_mask is not None:
            fit = out.loc[train_mask.reindex(out.index).fillna(False).astype(bool)]
        else:
            fit = out
        for c in cols:
            src = fit[c] if not fit[c].dropna().empty else out[c]
            lo, hi = src.quantile(p), src.quantile(1 - p)
            out[c] = out[c].clip(lo, hi)
        return out
    raise ValueError(f'unknown tame action {action!r}')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features_scaling.py -k tame -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/irp/features/engineering.py tests/test_features_scaling.py
git commit -m "feat: add tame_columns (clip/log/drop) for heavy-tailed features"
```

---

### Task 3: `residual_scale_flags`

**Files:**
- Modify: `src/irp/features/engineering.py` (add after `tame_columns`)
- Test: `tests/test_features_scaling.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features_scaling.py`:

```python
def test_residual_scale_flags_fires_then_silent_after_clip():
    rng = np.random.default_rng(1)
    n = 1000
    df = pd.DataFrame({
        'Date': [datetime.date(2020, 1, 1)] * n,
        'Ticker': [f'T{i}' for i in range(n)],
        'huge': rng.standard_cauchy(n) * 1e6,
        'small': rng.uniform(-1, 1, n),
    })
    flags = eng.residual_scale_flags(df, ['huge', 'small'])
    assert 'huge' in flags and 'small' not in flags
    clipped = eng.tame_columns(df, ['huge'], 'clip', p=0.01)
    assert eng.residual_scale_flags(clipped, ['huge', 'small']) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features_scaling.py -k residual -v`
Expected: FAIL — `has no attribute 'residual_scale_flags'`.

- [ ] **Step 3: Write minimal implementation**

In `src/irp/features/engineering.py`, after `tame_columns`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features_scaling.py -k residual -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/irp/features/engineering.py tests/test_features_scaling.py
git commit -m "feat: add residual_scale_flags post-scale check"
```

---

### Task 4: Wire detect/tame/residual into `_split_and_scale`

**Files:**
- Modify: `src/irp/ui/services/features_service.py` (`ScaleCfg` ~line 32; `_split_and_scale` ~line 149-163)
- Test: `tests/test_features_service.py` (create if absent; otherwise append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_features_service.py` (create the file with the imports if it does not exist):

```python
import datetime

import numpy as np
import pandas as pd

from irp.ui.services import features_service as svc


def _panel_with_fat_col():
    rng = np.random.default_rng(2)
    rows = []
    for y in (2018, 2020):
        d = datetime.date(y, 12, 31)
        for i in range(200):
            rows.append({'Date': d, 'Ticker': f'T{i}',
                         'roe': rng.standard_cauchy() * 1e6,
                         'fwd_ret': 0.0, 'label': 0})
    return pd.DataFrame(rows)


def test_split_and_scale_tames_then_scales_with_notes():
    panel = _panel_with_fat_col()
    scale_cfg = {'method': 'robust', 'scope': 'date',
                 'tame_action': 'clip', 'tame_cols': ['roe'], 'tame_p': 0.01}
    out, notes = svc._split_and_scale(panel, scale_cfg, None)
    # detection named the fat column up front
    assert any('roe' in n for n in notes)
    # after clip + robust-per-date scaling, no residual warning remains
    assert not any('still large' in n for n in notes)
    assert out['roe'].abs().quantile(0.99) < 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features_service.py -k tames_then_scales -v`
Expected: FAIL — notes do not yet mention `roe` (detect/tame not wired).

- [ ] **Step 3: Extend `ScaleCfg`**

In `src/irp/ui/services/features_service.py`, replace the `ScaleCfg` body (lines 32-37):

```python
class ScaleCfg(TypedDict, total=False):
    """Model-preprocessing scaling spec."""
    method: str          # none|minmax|robust
    scope: str           # date|global|ticker
    train_cutoff: int    # fit-year cutoff for global/ticker scope
    tame_action: str     # none|clip|log|drop — heavy-tail handling pre-scale
    tame_cols: list[str] # columns the tame action targets
    tame_p: float        # clip tail fraction
```

- [ ] **Step 4: Wire detect/tame/residual into `_split_and_scale`**

In `src/irp/ui/services/features_service.py`, replace the scaling block (the `if method and method != 'none':` body, lines 149-163) with:

```python
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
            # per-date scaling fits within each date; clip then shares that train_mask
            tame_mask = train_mask
            if tame_mask is None and cutoff is not None:
                tame_mask = pd.to_datetime(panel['Date']).dt.year <= int(cutoff)

            heavy = _eng.detect_heavy_tailed(panel, feat_cols)
            if heavy:
                notes.append('heavy-tailed: ' + ', '.join(heavy) +
                             ' — clip/log/drop them in the Scaling controls')

            tame_action = scale_cfg.get('tame_action', 'none')
            tame_cols = scale_cfg.get('tame_cols') or []
            if tame_action and tame_action != 'none' and tame_cols:
                panel = _eng.tame_columns(panel, tame_cols, tame_action,
                                          p=float(scale_cfg.get('tame_p', 0.01)),
                                          train_mask=tame_mask)
                feat_cols = [c for c in panel.columns if c not in _RESERVED_COLS]

            panel = _eng.scale_features(panel, feat_cols, method=method, scope=scope,
                                        train_cutoff=cutoff, train_mask=train_mask)

            residual = _eng.residual_scale_flags(panel, feat_cols)
            if residual:
                notes.append('still large after scaling: ' +
                             ', '.join(f'{c} (p99={v:.3g})' for c, v in residual.items()))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_features_service.py -k tames_then_scales -v`
Expected: PASS.

- [ ] **Step 6: Run the full scaling + service suites**

Run: `uv run pytest tests/test_features_scaling.py tests/test_features_service.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/irp/ui/services/features_service.py tests/test_features_service.py
git commit -m "feat: detect/tame/warn heavy-tailed cols in features scaling stage"
```

---

### Task 5: `/features` UI controls + wiring

**Files:**
- Modify: `src/irp/ui/pages/features.py` (3b block ~line 315-324; `update_col_options` ~line 410; toggles ~line 388-394; `_spec` ~592; `run_build` ~647; `save_recipe` ~771; `load_recipe_controls` ~803)

- [ ] **Step 1: Add the tame control-row to the 3b block**

In `src/irp/ui/pages/features.py`, immediately after the existing scaling `control-row` (closes at line 324, before the `# ── Step 3c` comment), insert:

```python
        html.P('Heavy-tailed columns (named in build notes after a build) stay huge '
               'after robust/minmax. Pick them here to clip / log before scaling, or '
               'drop them. Clip & Log fit on the train rows.',
               style={'color': MUTED, 'fontSize': '12px', 'margin': '6px 0 4px'}),
        html.Div(className='control-row', style={'alignItems': 'flex-end'},
                 id='feat-tame-row', children=[
            _field('Heavy tails',
                   dcc.RadioItems(id='feat-scale-tame-action',
                                  options=[{'label': 'None', 'value': 'none'},
                                           {'label': 'Clip', 'value': 'clip'},
                                           {'label': 'Log', 'value': 'log'},
                                           {'label': 'Drop', 'value': 'drop'}],
                                  value='none', inline=True, labelClassName='check-item')),
            _field('Columns',
                   dcc.Dropdown(id='feat-scale-tame-cols', options=[], value=[],
                                multi=True, placeholder='columns to tame…',
                                style={'width': '320px'})),
            _field('Clip p', _num('feat-scale-tame-p', '0.01', 0.01, '90px'),
                   wrap_id='feat-f-tame-p'),
        ]),
```

- [ ] **Step 2: Add tame-cols to the column-options callback**

In `update_col_options` (line 411), add the third output. Replace the decorator line and body:

```python
@callback(
    Output('feat-col-a', 'options'), Output('feat-col-b', 'options'),
    Output('feat-scale-tame-cols', 'options'),
    Input('feat-freq', 'value'),
)
def update_col_options(freq):
    """Dense frequencies expose the fundamentals-only + price/volume palette."""
    opts = _COL_OPTIONS_DENSE if freq in _DENSE_FREQS else _COL_OPTIONS
    return opts, opts, opts
```

- [ ] **Step 3: Add the tame-visibility callback**

After `toggle_scale_cutoff` (ends line 394), add:

```python
@callback(
    Output('feat-tame-row', 'style'),
    Output('feat-f-tame-p', 'style'),
    Input('feat-scale-method', 'value'),
    Input('feat-scale-tame-action', 'value'),
)
def toggle_tame_inputs(method, action):
    """Tame row only when scaling is on; Clip p only for the clip action."""
    row = dict(_HIDE) if method in (None, 'none') else {'display': 'flex'}
    p = dict(_SHOW) if action == 'clip' else dict(_HIDE)
    return row, p
```

- [ ] **Step 4: Thread the 3 values through `_spec`**

In `_spec` (line 592), add params and write them into the `scale` dict. Replace the signature line and the `'scale':` entry:

```python
def _spec(start, end, freq, variant, steps, horizon, mode, buckets,
          market, sector, watchlist, scale_method='none', scale_scope='date',
          scale_cutoff=None, tame_action='none', tame_cols=None, tame_p=0.01,
          split_method='none', split_train=0.7, split_valid=0.15,
          split_test=0.15, split_seed=0) -> dict:
```

and:

```python
        'scale': {'method': scale_method or 'none', 'scope': scale_scope or 'date',
                  'train_cutoff': int(scale_cutoff) if scale_cutoff not in (None, '') else None,
                  'tame_action': tame_action or 'none', 'tame_cols': tame_cols or [],
                  'tame_p': float(tame_p) if tame_p not in (None, '') else 0.01},
```

- [ ] **Step 5: Add States + params to `run_build`**

In the `run_build` callback (line 647), after the `feat-scale-cutoff` State (line 660) add:

```python
    State('feat-scale-tame-action', 'value'), State('feat-scale-tame-cols', 'value'),
    State('feat-scale-tame-p', 'value'),
```

Update the signature (line 668) to insert the 3 params after `scale_cutoff`:

```python
def run_build(n, start, end, freq, variant, steps, horizon, mode, buckets,
              market, sector, watchlist, scale_method, scale_scope, scale_cutoff,
              tame_action, tame_cols, tame_p,
              split_method, split_train, split_valid, split_test, split_seed):
```

and the `_spec(...)` call (line 671) to pass them in the same position:

```python
    spec = _spec(start, end, freq, variant, steps, horizon, mode, buckets,
                 market, sector, watchlist, scale_method, scale_scope, scale_cutoff,
                 tame_action, tame_cols, tame_p,
                 split_method, split_train, split_valid, split_test, split_seed)
```

- [ ] **Step 6: Add States + params to `save_recipe`**

In `save_recipe` (line 771), after the `feat-scale-cutoff` State (line 784) add the same 3 States:

```python
    State('feat-scale-tame-action', 'value'), State('feat-scale-tame-cols', 'value'),
    State('feat-scale-tame-p', 'value'),
```

Update its signature (line 791) and `_spec(...)` call (line 796) identically to Task 5 Step 5 (insert `tame_action, tame_cols, tame_p` after `scale_cutoff`):

```python
def save_recipe(n, name, start, end, freq, variant, steps, horizon, mode, buckets,
                market, sector, watchlist, scale_method, scale_scope, scale_cutoff,
                tame_action, tame_cols, tame_p,
                split_method, split_train, split_valid, split_test, split_seed, trigger):
```

```python
    spec = _spec(start, end, freq, variant, steps, horizon, mode, buckets,
                 market, sector, watchlist, scale_method, scale_scope, scale_cutoff,
                 tame_action, tame_cols, tame_p,
                 split_method, split_train, split_valid, split_test, split_seed)
```

- [ ] **Step 7: Add Outputs + restore in `load_recipe_controls`**

In `load_recipe_controls` (line 803), after the `feat-scale-cutoff` Output (line 811) add:

```python
    Output('feat-scale-tame-action', 'value'), Output('feat-scale-tame-cols', 'value'),
    Output('feat-scale-tame-p', 'value'),
```

In the return tuple (line 832), insert the three values right after `sc.get('train_cutoff')`:

```python
            sc.get('method', 'none'), sc.get('scope', 'date'), sc.get('train_cutoff'),
            sc.get('tame_action', 'none'), sc.get('tame_cols', []), sc.get('tame_p', 0.01),
```

- [ ] **Step 8: Lint + import-time smoke**

Run: `uv run ruff check src/irp/ui/pages/features.py src/irp/ui/services/features_service.py src/irp/features/engineering.py`
Expected: no errors.

Run: `uv run python -c "import irp.ui.pages.features"`
Expected: imports clean (Dash callback signatures match Outputs/States counts — a mismatch raises at import).

- [ ] **Step 9: Manual smoke**

Run: `uv run python -m irp.ui`, open `/features`:
- Pick a scale Method → the Heavy tails row appears; None → it hides.
- Set action = Clip → "Clip p" shows; Log/Drop/None → it hides.
- Build with `roe`/`close` selected and Clip → build note lists detected heavy-tailed cols, no "still large" note; export round-trips; Save then Load recipe restores action/cols/p.

- [ ] **Step 10: Commit**

```bash
git add src/irp/ui/pages/features.py
git commit -m "feat: heavy-tail clip/log/drop controls on /features scaling"
```

---

### Task 6: Docs (CLAUDE.md + md_scratchpad)

**Files:**
- Modify: `CLAUDE.md` (the `features_service.py` / `/features` entries describing scaling)
- Modify: `md_scratchpad/quant_research_overview.md`

- [ ] **Step 1: Update CLAUDE.md**

In the `features_service.py` paragraph, extend the scaling description to note the heavy-tail
stage: detection warning + `tame_action`/`tame_cols`/`tame_p` (clip fit on train, log =
signed-log, drop removes the column) running before `scale_features`, plus the residual
post-scale warning. In the `/features` page section, add the "Heavy tails" control-row
(action radio + columns multi-select + clip-p) to the Row 3b description.

- [ ] **Step 2: Update md_scratchpad/quant_research_overview.md**

Add a short subsection under the features/scaling walkthrough documenting: the heavy-tail
detection metric (tail-ratio (q99.9−q0.1)/IQR), the three actions, train-fit clip, and that
warnings appear in the build notes. Note the principle: all scaling/taming lives on
`/features`; notebooks never scale.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md md_scratchpad/quant_research_overview.md
git commit -m "docs: document heavy-tail taming on /features"
```

---

## Self-Review

**Spec coverage:**
- Pipeline detect→tame→scale→residual → Task 4. ✓
- `detect_heavy_tailed` → Task 1; `tame_columns` clip/log/drop train-fit → Task 2; `signed_log`
  → Task 1; `residual_scale_flags` → Task 3. ✓
- `ScaleCfg` extension → Task 4 Step 3. ✓
- UI controls + options + visibility + `_spec`/run/save/load threading → Task 5. ✓
- Warnings on the existing notes channel → Task 4 (notes), surfaced unchanged by `run_build`. ✓
- Tests for every function + service integration → Tasks 1-4. ✓
- Docs (both files, per project rule) → Task 6. ✓

**Placeholder scan:** none — every code step has full code; docs steps name exact files + content.

**Type consistency:** `detect_heavy_tailed(df, cols, thresh)`, `tame_columns(df, cols, action, p, train_mask)`, `residual_scale_flags(df, cols, thresh)`, `signed_log(s)` — signatures identical across plan and tests. `_TAME_RESERVED` defined in Task 2, reused in Task 3. `tame_action`/`tame_cols`/`tame_p` keys identical in `ScaleCfg`, `_split_and_scale`, `_spec`, and the recipe load. Param insertion position (after `scale_cutoff`, before `split_method`) consistent across `_spec`, `run_build`, `save_recipe`.
