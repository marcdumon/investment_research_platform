# Investment Research Platform

Modular research system for collecting, standardizing, and deriving signals from financial data.

## Layers

| Layer | Purpose |
|-------|---------|
| Sources | Fetch external data (prices, fundamentals) |
| Datasets | Unified internal representation |
| Transforms | Deterministic transforms (cleaning, alignment, joins) |
| Features | Derived signals — pure functions over datasets |
| Research | Hypothesis exploration |
| Evaluation | Signal scoring and backtesting |

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
```

## Configuration

Non-secret config lives in `config.toml`:

```toml
[simfin]
data_dir = "data/simfin"
```

Secrets go in `.env` (gitignored):

```
SIMFIN_API_KEY=
```

### Get a SimFin API key

Register at `https://simfin.com` and copy your API key from account settings.

## Price data (Stooq)

### Initial load — full history

1. Go to `https://stooq.com/db/h/` (captcha per file)
2. Download the zips you need:
   - `d_us_txt.zip` — US equities (NYSE, NASDAQ, etc.)
   - `d_world_txt.zip` — indices, FX, bonds, crypto, commodities
3. Place them in `data/stooq/download/`
4. Run:

```bash
uv run python scripts/fetch_stooq_prices.py
```

Each zip is extracted once to `data/stooq/daily/<zip_stem>/`. Re-downloading a zip with a newer mtime triggers automatic re-extraction on the next run.

### Incremental updates

1. Go to `https://stooq.com/db/`
2. Select the missing date range (up to ~1 month back)
3. Download — the file is named `data_d.txt`
4. Save it to `data/stooq/daily/data_d.txt` (overwrite)
5. Run:

```bash
uv run python scripts/update_stooq_prices.py
```

All rows are upserted on `(source_id, date)` — safe to re-run.

## Running tests

```bash
uv run pytest
uv run pytest -v
uv run pytest tests/test_store.py   # specific file
```
