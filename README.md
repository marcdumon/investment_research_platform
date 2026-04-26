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
STOOQ_API_KEY=
SIMFIN_API_KEY=
```

### Get a Stooq API key

Free, no registration required:

1. Go to `https://stooq.com/q/d/?s=msft.us&get_apikey`
2. Solve the captcha
3. Copy the key from the CSV download link shown at the bottom

### Stooq bulk data (optional)

For full historical data without per-ticker API calls, download bulk zips manually:

1. Go to `https://stooq.com/db/h/` (requires captcha per file)
2. Download the zips you need — recommended:
   - `d_us_txt.zip` — US equities (NYSE, NASDAQ, etc.)
   - `d_world_txt.zip` — indices, FX, bonds, crypto, money markets
3. Place them in `data/stooq/download/`

On first `StooqBulkSource.fetch()` each zip is extracted automatically to `data/stooq/daily/`.
Subsequent calls read from disk with no network request.

### Get a SimFin API key

Register at `https://simfin.com` and copy your API key from account settings.

## Running tests

```bash
uv run irp test          # run all tests
uv run irp test -v       # verbose
uv run irp test tests/sources/test_stooq.py  # specific file
```
