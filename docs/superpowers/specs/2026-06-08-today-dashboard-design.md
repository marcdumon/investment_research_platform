# Today decision dashboard (`/today`) — design

Sub-project #1 of 4 in the workflow-improvement effort (order: dashboard → shared context →
model→action → data refresh). Each ships independently.

## Context

The research loop is spread across islands (`/screener`, `/analysis`, `/regime`, `/backtest`,
`/features`). Nothing answers "what do I do today". This page ties the freshly-built regime +
factor-conditioning + composite ranking into one daily decision surface: read the macro
regime, see the playbook it implies, and get the top names ranked by the regime-appropriate
signal — plus your watchlist's standing.

Self-contained for now (its own controls). Sub-project #2 (shared context) later makes its
rows click through to `/analysis` etc. without losing state.

## Architecture

```
pages/today.py  (/today, nav + home card; lands as the decision surface)
  └─ services/today_service.py     (composes existing services; thin)
       ├─ regime_service.current_regime()      (NEW: light rule label+score+drivers, no HMM)
       ├─ factors_service._load_cross_section(date, variant, market/sector/watchlist)
       ├─ features.composite.build_composite / PRESETS
       ├─ snapshot.latest_cached(variant)       (NEW: most recent cached cross-section date)
       └─ watchlist_service.list/load
  └─ charts.py / theme.py
```

## New pure/helper pieces (testable)

- `regime_service.current_regime(end=None) -> dict` — `feature_panel(None, end)` + `rule_regime`
  last row → `{label, score, drivers}` (drivers = `feature_contributions` top ±). Fast: no HMM.
- `irp.factors.cache.snapshot.latest_cached(variant) -> date | None` — max date among
  `data/factor_cache/<variant>/*.parquet`.
- `today_service._rank_by_composite(df, weights, n) -> DataFrame` — `build_composite` →
  `score` column → top-n by score, carrying Ticker/Company/Sector + the preset's factor cols.
  Pure (operates on a passed cross-section frame); unit-tested on a synthetic frame.

## Service: `today_service.py`

- `_REGIME_PLAYBOOK: dict[label, dict]` — transparent map:
  `risk_on → {preset:'momentum', stance:'Lean into trend; full gross'}`,
  `neutral → {preset:'composite', stance:'Balanced; normal gross'}`,
  `risk_off → {preset:'quality', stance:'Defensive; reduce gross'}`,
  `unknown → {preset:'composite', stance:'Insufficient regime history'}`.
- `dashboard(variant='A', market, sector, watchlist, top_n=20, preset=None) -> TodayResult`
  (dataclass): current regime dict + playbook + chosen preset (regime's if `preset=None`) +
  `as_of` date (latest cached) + `top` DataFrame (`_rank_by_composite` on the cross-section) +
  `warnings`. Cold cache → warning pointing at Precompute.
- `watchlist_panel(name, variant, preset) -> DataFrame` — watchlist tickers' composite score +
  percentile rank within the full cross-section + latest momentum (`mom_12_1` if present).
- `playbook_preset(label) -> str`, `regime_color(label) -> str` helpers for the page.

## Page: `/today`

- **Section A — Regime banner**: regime chip (colour by risk_on/neutral/risk_off) + risk score
  + top 3 drivers (from `current_regime`) + the playbook stance + chosen preset. Link to
  `/regime` for depth. `as_of` date shown.
- **Section B — Controls**: variant A/Q, market / sector / watchlist filters, top-N, a
  "Use regime preset" toggle (auto-pick from regime) vs a manual preset dropdown. Run.
- **Section C — Top names**: ranked DataTable by the chosen composite — Rank, Ticker, Company,
  Sector, Score + a few key factor columns. Sortable.
- **Section D — Watchlist standing**: if a watchlist is set, its names with composite score,
  universe percentile, momentum — quick "is my book aligned with the regime" read.
- Nav: add `Today` to `components.navbar` (first after Home) + a home card.

Cache/token render pattern copied from `pages/regime.py` (`_put` + token store + builder).

## Testing

- `tests/test_today_service.py`: `_rank_by_composite` ranks a planted top name first on a
  synthetic cross-section; `_REGIME_PLAYBOOK` maps each regime label to a valid `PRESETS` key;
  `playbook_preset('risk_off')=='quality'`.
- `regime_service.current_regime()` smoke: returns a label in {risk_on,neutral,risk_off,unknown}.
- `snapshot.latest_cached('A')` returns a date or None without raising.

## Reuse (do not reinvent)

- `factors_service._load_cross_section` (already joins Sector/Company/Market).
- `features.composite.build_composite` + `PRESETS`.
- `regime_service.feature_panel` / `rule_regime` / `feature_contributions`.
- `watchlist_service`; page helpers + `_put`/token pattern from `pages/regime.py`.

## Verification

- `uv run pytest tests/test_today_service.py tests/test_regime.py -q` green.
- `uv run ruff check src` + `uv run mypy` clean; `uv run python -c "import irp.ui.app"`.
- Headless: `today_service.dashboard()` returns a regime + top names frame; `_build_*` render
  functions construct.
- Manual: `/today` shows regime banner + top names; switching market/preset updates the table.
- Docs: CLAUDE.md (`/today` route + `today_service` + `current_regime`/`latest_cached`) +
  md_scratchpad Phase 18.
