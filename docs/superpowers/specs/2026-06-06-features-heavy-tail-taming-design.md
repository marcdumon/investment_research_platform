# Heavy-Tail Taming on /features

## Problem

The `/features` page scales feature columns (minmax or robust) before export. Neither
method clips outliers, so fat-tailed columns keep extreme magnitudes after scaling. Robust
scaling `(x − median) / IQR` rescales the bulk but leaves heavy tails huge; minmax-global
lets post-cutoff test rows blow past `[0, 1]`. Observed in a real export: `close` max `1e11`,
`roe` max `4.9e6`, `revenue` min `-6620`, `total_assets` max `6.4e4` — all "scaled" yet still
enormous. Downstream, this stalls lbfgs in the classifier notebook (ConvergenceWarning) and
silently degrades any linear model.

## Principles (non-negotiable)

1. All scaling / data transforms happen on `/features`. Notebooks never scale or modify data.
2. Every transform is an explicit user control and is named in the build notes. Nothing hidden.
3. Scaling and clip parameters are fit on the **train** rows and applied to valid/test.
4. When scaling fails to tame a column, surface a warning (before and after scaling).
5. The user selects which columns to tame and the action (clip / log / drop).

## Scope

One feature: a "heavy tails" control inside the existing Scaling block, plus the engineering
functions and warnings that back it. Single implementation plan. Out of scope: changing the
existing scale methods, the notebook, or the manual `winsorize` step op (untouched).

## Pipeline

Heavy-tail handling slots into `_split_and_scale` (`features_service.py`), which already runs
last (after steps + label) and already emits `notes`:

```
split → DETECT heavy-tail (warn) → TAME selected cols (train-fit) → scale → RESIDUAL check (warn)
```

- **DETECT** runs over the feature columns, names heavy-tailed ones in the notes.
- **TAME** applies the user's chosen action to the user-selected columns, before the scaler.
  Clip params are fit on the same train rows the scaler uses (the `train_mask` already
  computed in `_split_and_scale`). Drop removes the column. Log is stateless.
- **RESIDUAL** runs after `scale_features`, names any feature column whose magnitude is still
  large in the notes.

All warnings are non-blocking and append to the `notes` list, which `build_panel` returns as
`missing + snotes` and the page surfaces in the build-status "Note:" line. Drop is the
remove-the-column path the user asked for.

## Engineering layer (`src/irp/features/engineering.py`) — pure, TDD'd

### `detect_heavy_tailed(df, cols, thresh=20.0) -> list[str]`
Flags columns whose tail-ratio `(q(0.999) − q(0.001)) / (q(0.75) − q(0.25))` exceeds `thresh`.
NaN-skipping. Columns with zero IQR (constant) are not flagged. Bounded columns (rsi in
`[-7,7]`, rand in `[-1,1]`) fall below the threshold and are ignored.

### `tame_columns(df, cols, action, p=0.01, train_mask=None) -> df`
Applies one `action` to every column in `cols` (intersected with `df.columns`):
- `clip`: winsorize to `[p, 1−p]` quantiles **computed on `train_mask` rows** (full sample if
  no mask), applied to all rows in place. Leak-free w.r.t. the test split.
- `log`: signed-log `sign(x) · log1p(|x|)` in place. Stateless, monotonic, handles negatives
  and zero (`log1p(0) = 0`). Train/test consistent by construction.
- `drop`: remove the columns from the frame.
- `none` / empty `cols`: return `df` unchanged.
Reserved columns (`Date/Ticker/fwd_ret/label/split`) are never touched even if passed in.

### `residual_scale_flags(df, cols, thresh=10.0) -> dict[str, float]`
After scaling, returns `{col: p99_abs}` for each feature column whose 99th-percentile absolute
value exceeds `thresh`. NaN-skipping. Empty dict when all columns are tamed.

`signed_log(s) -> pd.Series` may be a small shared helper used by `tame_columns`.

## Config (`features_service.py`)

`ScaleCfg` TypedDict gains:
- `tame_action: str` — `none | clip | log | drop` (default `none`)
- `tame_cols: list[str]` — columns to tame (default `[]`)
- `tame_p: float` — clip tail fraction (default `0.01`)

`_split_and_scale` reads these from `scale_cfg`. The detect/tame/residual calls wrap the
existing `scale_features` call. Tame and scale share the one `train_mask` already derived from
the split (or the cutoff). `build_panel` is unchanged — it already forwards
`scale_cfg=spec.get('scale')`.

Detection always runs when scaling is on (to warn), regardless of `tame_action`. Residual check
always runs after scaling. Tame only mutates columns when `tame_action != none` and `tame_cols`
is non-empty.

## UI (`src/irp/ui/pages/features.py`)

A second control-row in the "3b · Scale features" block:
- `feat-scale-tame-action` — `RadioItems` None / Clip / Log / Drop (value `none`).
- `feat-scale-tame-cols` — `Dropdown` `multi=True`; options = feature columns.
- `feat-scale-tame-p` — numeric, wrap id `feat-f-tame-p`, default `0.01`.

Helper text: "Heavy-tailed columns (named in build notes) stay huge after robust/minmax. Pick
them here to clip / log before scaling, or drop them. Clip & Log fit on the train rows."

**Options population:** add `Output('feat-scale-tame-cols', 'options')` to the existing
freq-driven column-options callback (currently sets `feat-col-a` / `feat-col-b` from
`available_columns(freq)`).

**Visibility:** new callback `toggle_tame_inputs(method, action)` — hide the whole tame row when
`method == 'none'`; show `feat-f-tame-p` only when `action == 'clip'`. Inputs: `feat-scale-method`,
`feat-scale-tame-action`.

**Thread through:**
- `_spec(...)` — new params `tame_action='none', tame_cols=None, tame_p=0.01`; written into the
  `'scale'` dict as `tame_action`, `tame_cols: tame_cols or []`, `tame_p: float(tame_p or 0.01)`.
- `run_build` — +3 `State`s, +3 params, pass to `_spec`.
- `save_recipe` — +3 `State`s, +3 params, pass to `_spec`.
- `load_recipe_controls` — +3 `Output`s, read `sc.get('tame_action','none')`,
  `sc.get('tame_cols', [])`, `sc.get('tame_p', 0.01)`.

Recipe JSON carries the tame keys inside `scale`, so save/load round-trips.

## Testing (TDD, `tests/test_features_scaling.py`)

- `detect_heavy_tailed` flags a fat-tailed column, ignores a bounded one, ignores constant.
- `tame_columns` clip: test rows clipped to **train** quantiles (perturbing test rows past the
  train cap does not change the cap — no leak).
- `tame_columns` log: `signed_log` is monotonic, `log1p(0)=0`, negatives map to negative output.
- `tame_columns` drop: column removed; reserved columns never dropped.
- `residual_scale_flags`: fires for a still-huge column, empty when the column was clipped first.
- Integration via `_split_and_scale` (service test): a fat-tailed column + `tame_action='clip'`
  appends a detection note and yields a tamed, scaled column; residual note absent.

## Verification

- `uv run pytest tests/test_features_scaling.py` green (new + existing).
- `uv run ruff check src` clean; `uv run mypy` clean for touched modules.
- Smoke `uv run python -m irp.ui`, open `/features`: tame row appears when a scale method is
  chosen, `Clip p` shows only for Clip, build with `roe`/`close` clipped → build note lists the
  detected cols and no residual warning; export round-trips; save/load recipe preserves the tame
  settings.
