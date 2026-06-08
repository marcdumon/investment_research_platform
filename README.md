# Investment Research Platform

DuckDB-backed data pipeline for equity fundamentals and price data. Three providers: **SimFin** (fundamentals, company metadata), **Stooq** (historical OHLCV prices, bulk snapshot), **Yahoo** (corporate actions: dividends + splits, and live-adjusted OHLCV prices).

> **Non-expert user guide:** [`docs/user_manual.md`](docs/user_manual.md) — plain-language walkthrough of the **Feature Engineering** workflow (`/features` + `/feature-engineering`) and the **Regime** page (`/regime`), explaining every section and chart and how to use them for investment decisions.

---

## Datasets

### SimFin

| Dataset | Tables in DB | Variants | Markets |
|---|---|---|---|
| Fundamentals | `income`, `balance`, `cashflow` | `annual`, `quarterly` | `us`, `de` |
| Companies | `companies` | — | `us`, `de` |
| Meta | — | — | — |

Fundamentals follow **SEC period convention**: annual periods named by the calendar year the fiscal year ends (e.g. `2023FY`), derived from `Report Date`.

### Stooq

| Dataset | Table in DB | Notes |
|---|---|---|
| Bulk historical prices | `prices` | Daily OHLCV for US + World markets |
| Daily update | `prices` | Incremental OHLCV update (upserted) |

### Yahoo

| Dataset | Table in DB | Notes |
|---|---|---|
| Dividends | `dividends` | Per-ticker cash dividend events `(Ticker, Date, Amount)` |
| Splits | `splits` | Per-ticker stock-split events `(Ticker, Date, Ratio)` |
| Prices | `yahoo_prices` | Per-ticker auto-adjusted OHLCV `(Ticker, Date, Open, High, Low, Close, Volume)` |

All `Date` columns across `prices`, `dividends`, `splits`, and `yahoo_prices` are stored as DuckDB `DATE` type (`YYYY-MM-DD`). SimFin date columns (`Report Date`, `Publish Date`, `Restated Date`) are also `DATE`.

Source filter: `universe.yahoo_ticker IS NOT NULL` and `Market NOT IN config.providers.yahoo.markets_exclude` (default excludes: cryptocurrencies, money market, bonds → ~14k tickers). The `yahoo_ticker` column holds the yfinance-compatible symbol (e.g. `EURUSD=X` for currencies, `RNR-PF` for preferred shares); instruments with no Yahoo equivalent have `yahoo_ticker = NULL` and are skipped automatically.

---

## Configuration

`config.toml` at project root. Secrets in `.env` (gitignored):

```
SIMFIN_API_KEY=your_key_here
```

### Yahoo sleep settings

Three independent rate-limit knobs under `[providers.yahoo]`:

| Key | Default | Applied to |
|---|---|---|
| `batch_sleep` | `1.0` | After each `yf.download()` batch (one request per ~50 tickers) |
| `actions_sleep` | `0.5` | After each `yf.Ticker(t).actions` call (one request per ticker) |
| `ticker_sleep` | `0.5` | After each `yf.Ticker(t).history()` call in `prices_mode='ticker'` (one request per ticker) |

Batch mode hits Yahoo once per 50 tickers so `batch_sleep` can be shorter relative to the per-ticker modes. Increase any of these if you hit rate limits.

---

## How to Fetch Data

### SimFin — Initial Bulk Load

SimFin data is fetched via the SimFin API (requires API key in `.env`).

```bash
uv run python -m irp.runner
```

This runs the full pipeline: fetch → transform → store → cleanup for both providers.

To run SimFin only:

```python
from irp.ingest.sim_fin import SimFinSource
from irp.runner import load_data

src = SimFinSource()
load_data(src, 'bulk')
src.cleanup()
```

### Stooq — Initial Bulk Load

Stooq does **not** have a download API. Files must be manually downloaded.

**Step 1 — Download bulk zips** from http://stooq.com/db/h/:
- US, Daily, Ascii → `d_us_txt.zip`
- World, Daily, Ascii → `d_world_txt.zip`

Place both files in `data/stooq/raw/`.

**Step 2 — Run bulk load:**

```bash
uv run python -m irp.runner
```

Or Stooq only:

```python
from irp.ingest.stooq import StooqSource
from irp.runner import load_data

src = StooqSource()
load_data(src, 'bulk')
src.cleanup()
```

### Stooq — Daily Update

**Step 1 — Download update file** from http://stooq.com/db/:
1. Click `Setting Files Content`
2. Select `World` and `U.S.` markets → `Save configuration` → `Close`
3. Select update days or click `All days Select`
4. Click `N_d` to download `data_d.txt`

Place `data_d.txt` in `data/stooq/raw/`.

**Step 2 — Run update:**

```python
src = StooqSource()
load_data(src, 'update')
src.cleanup()
```

### Yahoo — Initial Bulk Load

Yahoo uses the `yfinance` API. No manual downloads. Reads target tickers from the `universe` table via the `yahoo_ticker` column — run the `universe` CLI step first to populate it (requires Stooq bulk to have been fetched).

`YahooSource` fetches two feeds per ticker (both default on):
- **actions** — dividends + splits via `yf.Ticker(t).actions` (full history per ticker)
- **prices** — auto-adjusted OHLCV via `yf.Ticker(t).history(period='max', auto_adjust=True)`, or batched via `yf.download()` when `prices_mode='batch'` (default, ~10× faster)

**Slow:** ~14k tickers, several hours end-to-end. **Resume-safe** — stop with Ctrl-C any time, rerun continues from where it left off.

```python
from irp.ingest.yahoo import YahooSource

src = YahooSource()
src.fetch_bulk()
src.transform('bulk')
src.store('bulk')
```

Or via the interactive CLI: `uv run irp` → tick `yahoo`.

Resume state lives in `data/yahoo/raw/`:
- `queried_actions.json` — tickers whose dividends/splits have been pulled
- `queried_prices.json` — tickers whose OHLCV history has been pulled (separate so partial runs resume only what is missing)
- `error_tickers.json` — tickers that errored (shared across feeds; a failing ticker is skipped for both feeds on the next run)
- `actions.csv` — long-format dividend + split rows
- `prices.csv` — long-format OHLCV rows

These JSON files are the **live source of truth** for fetch progress. The `catalog` table in DuckDB mirrors them as `yahoo_prices_queried`, `yahoo_prices_error`, `yahoo_actions_queried`, `yahoo_actions_error` boolean columns, but those are a snapshot written at catalog rebuild time, not live state.

To re-probe known errors or force a full refresh: `_fetch_ticker_data(skip_errors=False, skip_queried=False)`.

### Yahoo — Daily Update

```python
src = YahooSource()
src.update()
src.transform('update')
src.store('update')
```

`update()` is **incremental**: it queries `yahoo_prices` for the last stored date per ticker and only fetches rows after that date, with a 30-day lookback window applied so recently adjusted prices are refreshed. New tickers (not yet in the DB) get full history. Batches use the minimum last date of the group as the shared start date.

Actions (dividends + splits) always re-fetch full history — yfinance has no incremental endpoint for these, but the volume is small and the merge deduplicates on `(Ticker, Date)`.

---

## Ticker Universe (universe table)

The `universe` table is provider-agnostic — it is **not** a Stooq output. It holds one row per instrument:

| Column | Notes |
|---|---|
| `Ticker` | Canonical key — Stooq SrcId stripped of exchange suffix (e.g. `AAPL`) |
| `Market` | Market category from Stooq zip structure |
| `stooq_ticker` | Full Stooq source ticker / SrcId (e.g. `AAPL.US`); NULL if not Stooq-sourced |
| `yahoo_ticker` | yfinance symbol (e.g. `EURUSD=X`); NULL if no Yahoo equivalent |

`yahoo_ticker` translation rules applied at seed time:

| Translation rule | Example |
|---|---|
| Currencies, 6-char alpha | `EURUSD` → `EURUSD=X` |
| Currencies, non-standard (`NOK_I`, `EUR_I`) | `NULL` — Stooq-specific, no Yahoo equivalent |
| Stooq stocks indices (`^_UK`, `^_US`) | `NULL` — Stooq-proprietary basket indices |
| Preferred/series shares `BASE_X` | `RNR_F` → `RNR-PF` |
| All others | unchanged |

Two CLI steps (both require Stooq bulk fetch to have run):
- `seed-universe` — reads `data/stooq/raw/markets.csv`, applies translations, writes `data/universe.csv`
- `universe` — reads `data/universe.csv`, writes the DB table

`data/universe.csv` is the editable master file. Edit it to add/remove tickers or fix translations.
To re-seed from a new Stooq bulk download, tick `seed-universe` again.

```python
from irp.query.universe import universe, seed, refresh
df = universe()                 # all tickers
df = universe('AAPL')           # single ticker
seed()                          # rebuild universe.csv from Stooq markets.csv
refresh()                       # write DB table from universe.csv
```

---

## Data Catalog

`irp.query.catalog.catalog()` returns a single DataFrame with one row per ticker and columns covering every data source:

| Column group | Source |
|---|---|
| `stooq_first/last/rows` | `prices` table |
| `yahoo_first/last/rows` | `yahoo_prices` table |
| `yahoo_prices_queried/error` | `queried_prices.json` / `error_tickers.json` |
| `yahoo_actions_queried/error` | `queried_actions.json` / `error_tickers.json` |
| `div_count/first/last`, `split_count` | `dividends`, `splits` tables |
| `income_A/Q`, `balance_A/Q`, `cashflow_A/Q` | SimFin fundamental tables |
| `in_companies` | `companies` table |

Rebuild via `uv run irp` → Steps → `catalog`. This reads the JSON files from `data/yahoo/raw/` and joins them with the current DB state; the resulting boolean columns are a snapshot, not live state.

```python
from irp.query.catalog import catalog
df = catalog()                 # all tickers
df = catalog('AAPL')           # single ticker
df = catalog(['AAPL', 'MSFT']) # subset
```

---

## Pipeline Internals

Each provider implements the same protocol:

```
fetch_bulk() / update()  →  transform(feed)  →  store(feed)  →  cleanup()
```

Steps are **idempotent**: freshness markers (`.fetched`, `.transformed_bulk`, etc.) in `data/<provider>/raw/` prevent redundant work. `cleanup()` deletes intermediate processed files to save disk space; markers and raw downloaded files are kept.

### Source modules

| Module | Description |
|---|---|
| `src/irp/ingest/sim_fin.py` | SimFin fetch, transform, store, update, cleanup |
| `src/irp/ingest/stooq.py` | Stooq unzip, transform, store, cleanup |
| `src/irp/ingest/yahoo.py` | Yahoo per-ticker dividends + splits + OHLCV via yfinance, resume-safe |
| `src/irp/runner.py` | Orchestrates providers via `DataProvider` protocol |
| `src/irp/core/freshness.py` | `is_fresh(marker, *inputs)` — skip logic |
| `src/irp/core/config.py` | Loads `config.toml` via Pydantic |

### Data directories

```
data/
  irp.duckdb              # single database file
  simfin/
    raw/                  # SimFin downloaded CSVs + freshness markers
      download/           # place manually downloaded zips here
    processed/            # intermediate CSVs (deleted by cleanup)
  stooq/
    raw/                  # Stooq zips + extracted data + freshness markers
    processed/            # intermediate files (deleted by cleanup)
  yahoo/
    raw/                  # actions.csv + prices.csv + queried_actions.json + queried_prices.json + error_tickers.json
    processed/            # dividends.csv + splits.csv + prices.csv (deleted by cleanup)
```

---

## Panel Layer

`irp.panel` materializes wide-format parquet panels from DuckDB for hot-path factor computation. All factor work runs here — pure polars/numpy, no SQL at query time.

```
data/panel/
  prices.parquet           # long-format (Ticker, Date, Close, Volume), loaded as dense float32 matrix
  income_A.parquet         # PIT-aligned income statement, annual
  income_Q.parquet
  balance_A.parquet / balance_Q.parquet
  cashflow_A.parquet / cashflow_Q.parquet
```

Rebuild after each ETL refresh via the Ingest page (*rebuild panel*) or:

```bash
uv run python -m irp.panel.build
```

The price panel (~1.2 GB, 25K dates × 12K tickers) is loaded once per process and cached in-memory. Factor decay over 10 years runs in ~3s warm vs ~25 minutes with the old SQL engine.

---

## Factors Analysis

`irp.factors` computes quant factors from stored fundamentals and prices. All results are point-in-time (PIT) safe: only data with `Publish Date <= as_of_date` (falling back to `Report Date + 60 days` when Publish Date is unavailable) and prices with `Date <= as_of_date` are used.

### Factors computed

| Group | Factors |
|---|---|
| Size | `mktcap` ($B) |
| Fundamentals | `revenue`, `net_income`, `total_assets`, `total_equity`, `op_cashflow` (all $B) |
| Valuation | `pe`, `pb`, `ps`, `ev_ebitda`, `ev_ebit`, `ev_sales`, `fcf_yield` |
| Profitability | `gross_margin`, `op_margin`, `net_margin`, `roe`, `roa`, `roic`, `fcf_margin`, `asset_turnover`, `cfo_ni_ratio`, `accruals` |
| Momentum | `mom_12_1`, `mom_6_1`, `vol_21d`, `ma200_ratio` |
| Leverage | `debt_equity`, `net_debt_ebitda`, `interest_coverage` |
| Growth | `rev_growth_1y`, `earn_growth_1y` |
| Quality | `piotroski_fscore` (0–9, 9 binary signals) |

**Momentum factor definitions:**

| Column | Formula | Notes |
|---|---|---|
| `mom_12_1` | log(P₋₃₀d / P₋₃₆₅d) | 12m return skipping last month (Jegadeesh-Titman) |
| `mom_6_1` | log(P₋₃₀d / P₋₁₈₂d) | 6m return skipping last month |
| `vol_21d` | std(daily log returns, 21d) × √252 | Annualised realised volatility |
| `ma200_ratio` | P₀ / SMA(Close, 200d) | Price relative to 200-day moving average |

Calendar-day lags approximate trading-day lags. Skipping the last month avoids short-term reversal bias.

### Adding a factor

Four steps — UI and normalization derive from the registry automatically. See `md_scratchpad/quant_research_overview.md` for the full walkthrough and table of available intermediate columns.

1. Formula in `_apply_formulas` in `irp/panel/cross_section.py`
2. `register('col', 'Label', pct=..., group=...)` in the relevant compute module
3. Append name to `_FACTOR_COLS_ORDER` in `cross_section.py`
4. Clear + rebuild the factor cache

### Usage

```python
import datetime
from irp.factors import cross_section, ticker_factor_history
from irp.factors.compute import run_backtest

# Full-universe cross-section at a point in time
df = cross_section(datetime.date(2024, 12, 31), variant='A')

# Factor history for one ticker (one row per annual filing)
df = ticker_factor_history('AAPL', variant='A')

# Quarterly: uses TTM (sum of last 4 quarters) for income/cashflow
df = cross_section(datetime.date(2024, 12, 31), variant='Q')

# Factor backtest: IC series + quintile cumulative returns
result = run_backtest('pe', horizon_days=252,
                      start_date=datetime.date(2018, 1, 1),
                      end_date=datetime.date(2024, 12, 31),
                      variant='A', freq='Q')
print(result['mean_ic'], result['ic_tstat'])
print(result['quintile_cumret'].tail())
```

`variant='Q'` routes income and cashflow through TTM aggregation (sum of last 4 quarterly filings) so ratios reflect a full year of activity rather than a single quarter. Balance sheet uses the most-recent quarter.

### Factor registry

Each compute module calls `register('col_name', 'UI Label', pct=True/False, group='...')` at import time, populating a central `irp.factors.registry`. UI labels, percentage formatting flags, and dropdown options are all derived from this registry — adding a new factor in one compute file automatically propagates to the UI and normalization layer with no other changes needed.

### Backtest module

`irp.factors.backtest` contains two pure functions (no DB access):

- `compute_forward_returns(prices, rebalance_dates, horizon_days)` — log return per ticker per date.
- `compute_backtest(factor, cross_sections, fwd_returns)` — Spearman IC series and equal-weight quintile cumulative returns.

`run_backtest()` in `compute.py` is the DB-layer entry point: fetches raw data once, generates rebalance dates, and calls both pure functions.

### Factor cache

`irp.factors.cache` persists full-universe cross-section snapshots to disk so they don't need recomputation on every app run. One parquet file per `(as_of_date, variant)` pair under `data/factor_cache/`.

Historical PIT snapshots are immutable, so the cache is valid until the underlying raw data changes. After a fresh ingest, clear the cache via the `/ingest` page (tick **clear factor cache**) or rebuild it in one shot (tick **rebuild factor cache**).

```python
from irp.factors import cache

cache.clear()                    # delete all cached snapshots
cache.clear('A')                 # delete only annual snapshots
n = cache.precompute_all(        # populate cache for a date range
    start_date=datetime.date(1991, 1, 1),
    end_date=datetime.date.today(),
)
```

---

## Features and Models

### Composite signals (`irp.features`)

`irp.features.normalize` provides cross-sectional normalization applied per snapshot date:

```python
from irp.features.normalize import zscore, rank_norm, sector_neutral

norm = zscore(df)                              # z-score, clipped at ±3
norm = rank_norm(df)                           # rank scaled to [-0.5, +0.5]
norm = sector_neutral(df, sector_series)       # demean within sector, then z-score
```

`irp.features.composite` combines normalized factors into a single composite score:

```python
from irp.features.composite import build_composite, PRESETS

# Use a predefined preset
score = build_composite(df, PRESETS['composite'])          # pe + pb + roe + roic + mom
score = build_composite(df, PRESETS['value'])
score = build_composite(df, PRESETS['quality'])
score = build_composite(df, PRESETS['momentum'])

# Custom weights (negative = short that signal)
score = build_composite(df, {'pe': -1, 'roe': 1, 'mom_12_1': 0.5}, normalize='rank')

# Sector-neutral composite
from irp.query.simfin import sector_map
score = build_composite(df, PRESETS['composite'], sector=sector_map())
```

Backtest a composite over a historical window:

```python
from irp.factors.compute import run_composite_backtest

result = run_composite_backtest(
    weights=PRESETS['composite'],
    horizon_days=252,
    start_date=datetime.date(2015, 1, 1),
    end_date=datetime.date(2024, 12, 31),
    normalize='zscore',
    use_sector_neutral=False,
)
print(result['mean_ic'], result['ic_tstat'])
```

### ML models (`irp.models`)

`irp.models.ml` runs walk-forward backtests using any sklearn-compatible estimator. Returns the same dict as `compute_backtest()` so results are directly comparable to single-factor and composite backtests.

```python
from sklearn.linear_model import Ridge
from irp.models.ml import run_ml_backtest
from irp.factors import cache
from irp.factors.backtest import compute_forward_returns
from irp.query.yahoo import prices as yahoo_prices
import pandas as pd, datetime

dates = [ts.date() for ts in pd.date_range('2015-01-01', '2024-12-31', freq='QE')]
xs = {d: cache.load(d, 'A') for d in dates}
xs = {d: v for d, v in xs.items() if v is not None}

prices = yahoo_prices(None)
fwd = compute_forward_returns(prices, list(xs.keys()), 252)

result = run_ml_backtest(Ridge(alpha=1.0), xs, fwd, n_train=20, n_test=4)
print(result['mean_ic'], result['ic_tstat'])
```

---

## Dash UI

```bash
uv run python -m irp.ui
```

Multi-page web UI. All pages linked in the navbar.

| Route | Description |
|---|---|
| `/` | Home |
| `/ingest` | Data ingestion — pipeline steps (fetch/transform/store/cleanup) + maintenance tasks (universe, catalog, panel rebuild, factor cache) |
| `/ticker` | Per-ticker fundamentals, prices, corporate actions |
| `/factors` | Cross-section screening (filterable by market/sector/date) + single-ticker factor history |
| `/backtest` | Single-factor and composite backtests — color-coded IC/t-stat chips, quintile cumulative returns, factor decay |
| `/screener` | Progressive filter stack on all factors + raw accounting items; scatter/histogram/price charts; lasso keep/remove; save as watchlist |
| `/correlation` | Factor-to-factor Pearson correlation heatmap (one cross-section snapshot) or ticker return-correlation heatmap (trailing window) |

---

## Development

```bash
uv run pytest                        # all tests
uv run pytest tests/test_store.py    # single file
uv run pytest -k "test_upsert"       # single test
```
