# Investment Research Platform

DuckDB-backed data pipeline for equity fundamentals and price data. Two providers: **SimFin** (fundamentals, prices, company metadata) and **Stooq** (historical OHLCV prices).

---

## Datasets

### SimFin

| Dataset | Tables in DB | Variants | Markets |
|---|---|---|---|
| Fundamentals | `income`, `balance`, `cashflow` | `annual`, `quarterly` | `us`, `de` |
| Share prices (Yahoo Finance) | `prices` | `daily`, `latest` | `us`, `de` |
| Dividends | `dividends` | — | `us`, `de` |
| Companies | `companies` | — | `us`, `de` |
| Meta | — | — | — |

Fundamentals follow **SEC period convention**: annual periods named by the calendar year the fiscal year ends (e.g. `2023FY`), derived from `Report Date`.

### Stooq

| Dataset | Table in DB | Notes |
|---|---|---|
| Bulk historical prices | `prices` | Daily OHLCV for US + World markets |
| Markets | `markets` | Ticker → market mapping |
| Daily update | `prices` | Incremental OHLCV update (upserted) |

---

## Configuration

`config.toml` at project root. Secrets in `.env` (gitignored):

```
SIMFIN_API_KEY=your_key_here
```

---

## How to Fetch Data

### SimFin — Initial Bulk Load

SimFin data is fetched via the SimFin API (requires API key in `.env`).

```bash
uv run python -m irp.pipeline
```

This runs the full pipeline: fetch → transform → store → cleanup for both providers.

To run SimFin only:

```python
from irp.sources.sim_fin import SimFinSource
from irp.pipeline import load_data

src = SimFinSource()
load_data(src, 'bulk')
src.cleanup()
```

### SimFin — Incremental Update (latest share prices only)

```python
src = SimFinSource()
load_data(src, 'update')
src.cleanup()
```

Only downloads `shareprices/latest` variant. Fundamentals and company data are refreshed on the `refresh_days` schedule configured in `config.toml` (default: 7 days).

### Stooq — Initial Bulk Load

Stooq does **not** have a download API. Files must be manually downloaded.

**Step 1 — Download bulk zips** from http://stooq.com/db/h/:
- US, Daily, Ascii → `d_us_txt.zip`
- World, Daily, Ascii → `d_world_txt.zip`

Place both files in `data/stooq/raw/`.

**Step 2 — Run bulk load:**

```bash
uv run python -m irp.pipeline
```

Or Stooq only:

```python
from irp.sources.stooq import StooqSource
from irp.pipeline import load_data

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
| `src/irp/sources/sim_fin.py` | SimFin fetch, transform, store, update, cleanup |
| `src/irp/sources/stooq.py` | Stooq unzip, transform, store, cleanup |
| `src/irp/pipeline.py` | Orchestrates both providers via `DataProvider` protocol |
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
```

---

## Development

```bash
uv run pytest                        # all tests
uv run pytest tests/test_store.py    # single file
uv run pytest -k "test_upsert"       # single test
```
