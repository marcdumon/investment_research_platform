# Architecture

```
External APIs / Files
  SimFin API ──────────────┐
  Stooq bulk zips ─────────┤  sources/
  SEC EDGAR API ───────────┘
          │
          ▼
      Dataset  ──►  transforms/  ──►  Store (DuckDB)
                                          │
                                          ▼
                                    anomalies/runner
                                          │
                             ┌────────────┼────────────┐
                         whitelist   blacklist    checklist
```

---

## `irp/config.py`

Loads `config.toml` from the project root.

| | |
|---|---|
| `Config` | TypedDict with `stooq`, `simfin`, `store` sections |
| `load()` | Parse and return `Config` |

---

## `irp/datasets/dataset.py`

Immutable data container passed between all layers.

| | |
|---|---|
| `Dataset` | Frozen dataclass: `name`, `data` (DataFrame), `schema`, `source`, `captured_at` |
| `Dataset.from_df()` | Create from DataFrame; infers schema from dtypes |
| `Dataset.evolve()` | Return copy with selected fields replaced |
| `Dataset.validate()` | Assert columns and dtypes match schema |

---

## `irp/store.py`

Persists and retrieves Datasets in a DuckDB file. Metadata (schema, source, PK) stored in `_irp_datasets`.

| | |
|---|---|
| `Store` | Main persistence class |
| `Store.save()` | Full overwrite (CREATE OR REPLACE) |
| `Store.upsert()` | Insert rows, update on PK conflict |
| `Store.load()` | Return Dataset by table name |
| `Store.delete()` | Drop table and remove metadata |
| `Store.exists()` | Check whether a table exists |
| `Store.get_max_date()` | Return MAX of a date column, optionally filtered by ticker |

---

## `irp/sources/`

Fetch raw data from external providers. All sources return a `Dataset`.

### `base.py`

| | |
|---|---|
| `BaseSource` | ABC; subclasses implement `fetch(**kwargs) -> Dataset` |

### `simfin.py`

| | |
|---|---|
| `SimFinFundamentalsSource` | Fetches income, balance, cashflow, industries, or companies from SimFin API |
| `SimFinFundamentalsSource.fetch()` | Download and return Dataset; adds `variant`, `period` (SEC convention), and `source` columns |

Period labels follow SEC convention: FY named by year it ends. Annual uses `Report Date` year; quarterly propagates Q4's year to Q1–Q3 of the same fiscal year.

### `stooq.py`

| | |
|---|---|
| `PRICE_SCHEMA` | Column-to-dtype mapping for the prices table |
| `normalize_ticker()` | Strip country suffix and uppercase: `"msft.us"` → `"MSFT"` |

### `stooq_bulk.py`

| | |
|---|---|
| `ensure_zips_extracted()` | Extract Stooq bulk zips into `data_dir`; skips already-extracted zips |
| `StooqBulkSource` | Reads OHLCV data for one ticker from extracted bulk zips |
| `StooqBulkSource.fetch()` | Locate ticker file, parse CSV, apply date filter, return Dataset |

### `sec_edgar.py`

| | |
|---|---|
| `get_sec_filing_url()` | Return `(url, form_type)` for a ticker + period from SEC EDGAR; uses `publish_date` hint for accurate matching |

---

## `irp/transforms/`

Stateless, chainable data transforms. Each takes a `Dataset` and returns a `Dataset`.

### `base.py`

| | |
|---|---|
| `Transformer` | ABC; subclasses implement `transform(dataset) -> Dataset` |

### `cleaner.py`

| | |
|---|---|
| `Cleaner` | Replace infinities with NaN; drop rows where all numeric columns are NaN; optionally percentile-clip |

### `caching.py`

| | |
|---|---|
| `CachingTransformer` | Wrap any Transformer; return cached Dataset from Store if available; otherwise run and save |

### `date_parser.py`

| | |
|---|---|
| `DateParser` | Parse a string date column to `datetime64` |

### `date_aligner.py`

| | |
|---|---|
| `DateAligner` | Reindex to a target date range; fill gaps via `ffill`, `bfill`, `nearest`, or `none`; per-ticker when `ticker_col` set |

### `forward_filler.py`

| | |
|---|---|
| `ForwardFiller` | Pivot to wide, reindex to daily, forward-fill per ticker, return to long format |

### `joiner.py`

| | |
|---|---|
| `Joiner` | Merge a fixed `right` Dataset into the input on `on` columns |

---

## `irp/anomalies/`

Detect data quality problems in fundamental tables.

### `base.py`

Core types shared by all rules.

| | |
|---|---|
| `Severity` | `ERROR` or `WARNING` |
| `Finding` | Schema dataclass for one anomaly: table, ticker, period, column, value, rule, detail, severity |
| `Finding.list_columns()` | Return ordered list of Finding field names |
| `Finding.empty_df()` | Return empty DataFrame with Finding columns |
| `Rule` | ABC; subclasses set `name`, `description`, `severity` and implement `check(data) -> DataFrame` |
| `Registry` | Holds (Rule class, config) pairs; instantiates rules on demand |
| `Registry.register` | Decorator to register a Rule class, with optional config overrides |
| `Registry.rules()` | Return list of instantiated Rule objects |
| `Registry.clear()` | Remove all registered rules |
| `default_registry` | Module-level Registry used by all `@register`-decorated rules |
| `register` | Alias for `default_registry.register` |

### `runner.py`

| | |
|---|---|
| `run()` | Run all registered rules against a dict of DataFrames; return findings enriched with `period`, company name, ISIN, and SEC filing links |

### `whitelist.py`

Known-good anomalies to suppress (e.g. intentional negative revenue quarter).

| | |
|---|---|
| `load()` | Load exceptions from `anomaly_whitelist.toml` |
| `suppress()` | Remove findings that match a whitelist entry |
| `add()` | Append one entry to the whitelist TOML |

### `blacklist.py`

Known-bad values that must always be flagged regardless of other rules.

| | |
|---|---|
| `load()` | Load entries from `anomaly_blacklist.toml` |
| `suppress()` | Filter findings to retain only blacklisted ones |
| `add()` | Append one entry to the blacklist TOML |

### `checklist.py`

Items under active investigation — tracked but not suppressed.

| | |
|---|---|
| `load()` | Load entries from `anomaly_checklist.toml` |
| `suppress()` | Remove findings that are already on the checklist |
| `add()` | Append one entry to the checklist TOML |

### `rules/`

| Rule class | `name` | What it checks |
|---|---|---|
| `ImpossibleValues` | `impossible_value` | Physically impossible values: negative revenue, non-positive assets/shares |
| `SuddenJumps` | `sudden_jump` | Period-over-period change exceeds ±threshold (default +500% / -80%) |
| `SectorOutliers` | `sector_outlier` | IQR-based cross-sectional outlier within industry group |
| `AccountingIdentity` | `accounting_identity` | Balance sheet: Assets = Liabilities + Equity; Income: Annual = Q1+Q2+Q3+Q4 |

---

## `irp/features/base.py`

| | |
|---|---|
| `Feature` | ABC for derived signals; subclasses set `name`, `required_datasets` and implement `compute(datasets) -> Series/DataFrame` |

---

## `irp/cli.py`

Entry point for the `irp` CLI tool.

| Command | Script function | What it does |
|---|---|---|
| `irp test` | `run_test()` | Run the test suite via pytest |

---

## `scripts/`

One-shot and incremental ETL scripts. Run with `uv run python scripts/<name>.py`.

| Script | Purpose |
|---|---|
| `fetch_simfin_fundamentals.py` | Initial load of income, balance, cashflow (annual + quarterly) |
| `update_simfin_fundamentals.py` | Force-refresh fundamentals from API |
| `fetch_stooq_prices.py` | Bulk load all Stooq prices from extracted zips |
| `update_stooq_prices.py` | Incremental price update from `data_d.txt` |
| `fetch_sec_filings.py` | Resolve SEC EDGAR filing URLs for all (ticker, period) pairs; per-ticker threading, skips already-resolved pairs |
| `patch_sec_filings_form.py` | One-off: infer `form` column from period string for existing rows (10-K/10-Q heuristic) |
| `verify_db.py` | Print row counts for all tables |
