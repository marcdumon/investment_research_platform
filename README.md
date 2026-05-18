# Investment Research Platform

DuckDB-backed data pipeline for equity fundamentals and price data. Three providers: **SimFin** (fundamentals, company metadata), **Stooq** (historical OHLCV prices, bulk snapshot), **Yahoo** (corporate actions: dividends + splits, and live-adjusted OHLCV prices).

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
| Markets | `markets` | Ticker → market mapping |
| Daily update | `prices` | Incremental OHLCV update (upserted) |

### Yahoo

| Dataset | Table in DB | Notes |
|---|---|---|
| Dividends | `dividends` | Per-ticker cash dividend events `(Ticker, Date, Amount)` |
| Splits | `splits` | Per-ticker stock-split events `(Ticker, Date, Ratio)` |
| Prices | `yahoo_prices` | Per-ticker live `auto_adjust=True` OHLCV `(Ticker, Date, Open, High, Low, Close, Volume)` |

Source filter: `markets` table minus `config.providers.yahoo.markets_exclude` (default: cryptocurrencies, money market, bonds → ~14k tickers).

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
uv run python -m irp.runner
```

This runs the full pipeline: fetch → transform → store → cleanup for both providers.

To run SimFin only:

```python
from irp.sources.sim_fin import SimFinSource
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
from irp.sources.stooq import StooqSource
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

Yahoo uses the `yfinance` API. No manual downloads. Reads target tickers from the `markets` table (run Stooq bulk first to populate it).

`YahooSource` can fetch two feeds per ticker (both default on):
- **actions** — dividends + splits via `yf.Ticker(t).actions`
- **prices** — live `auto_adjust=True` OHLCV via `yf.Ticker(t).history(period='max', auto_adjust=True)`

**Slow:** ~14k tickers × ~2× `batch_sleep` (one call per feed) → ~12h end-to-end for both. **Resume-safe** — stop with Ctrl-C any time, rerun continues.

```python
from irp.sources.yahoo import YahooSource

# both feeds (default)
src = YahooSource()
# or only one:
# src = YahooSource(fetch_actions=True, fetch_prices=False)
src.fetch_bulk()
src.transform('bulk')
src.store('bulk')
```

Or via the interactive CLI: `uv run irp` → tick `yahoo` → tick which content (actions, prices, or both).

Resume state lives in `data/yahoo/raw/`:
- `queried_actions.json` — tickers whose dividends/splits have been pulled
- `queried_prices.json` — tickers whose OHLCV history has been pulled (separate so partial runs resume only what is missing)
- `error_tickers.json` — tickers that errored on the yfinance call (shared: ticker-object failure is an error for both feeds)
- `actions.csv` — long-format dividend + split rows
- `prices.csv` — long-format OHLCV rows

To re-probe known errors or force a full refresh, call `_fetch_ticker_data(skip_errors=False, skip_queried=False)`.

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
| `src/irp/sources/yahoo.py` | Yahoo per-ticker dividends + splits + OHLCV via yfinance, resume-safe |
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

## Development

```bash
uv run pytest                        # all tests
uv run pytest tests/test_store.py    # single file
uv run pytest -k "test_upsert"       # single test
```
