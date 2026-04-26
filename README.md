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

### Get a SimFin API key

Register at `https://simfin.com` and copy your API key from account settings.

## Running tests

```bash
uv run pytest
```
