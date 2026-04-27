# Persistence Layer

Transformed datasets are saved to DuckDB so pipelines don't recompute from scratch every run.

## Storage

Single DuckDB file configured in `config.toml`:

```toml
[store]
db_path = "data/irp.duckdb"
```

Each dataset is stored as its own DuckDB table named after `dataset.name`.
Metadata (source, schema, captured_at) is tracked in an internal `_irp_datasets` table.

## API

```python
from irp.store import Store

store = Store()               # uses config.toml db_path
store = Store("custom.duckdb")  # override path
```

| Method | Description |
|--------|-------------|
| `store.save(dataset)` | Write dataset — overwrites if name exists |
| `store.load(name)` | Read dataset by name → `Dataset` |
| `store.list()` | List all saved dataset names |
| `store.delete(name)` | Drop dataset and its metadata |
| `store.exists(name)` | Check if dataset exists |

## CachingTransformer

Wraps any `Transformer`. On first call: runs the transform and saves the result.
On subsequent calls: returns the cached dataset from the store directly.

```python
from irp.transforms.caching import CachingTransformer

CachingTransformer(transformer, store, name, force=False)
```

| Param | Description |
|-------|-------------|
| `transformer` | Any `Transformer` instance |
| `store` | `Store` instance |
| `name` | Name under which to save/load the result |
| `force` | If `True`, always rerun even if cached |

## Pipeline with caching

```python
from irp.store import Store
from irp.sources.stooq_bulk import StooqBulkSource
from irp.transforms.date_parser import DateParser
from irp.transforms.cleaner import Cleaner
from irp.transforms.caching import CachingTransformer

store = Store()
raw = StooqBulkSource("msft.us").fetch()

prices = CachingTransformer(DateParser("date"), store, "prices_msft_parsed").transform(raw)
prices = CachingTransformer(Cleaner(),          store, "prices_msft_clean").transform(prices)

# second run: both steps load from DuckDB, no recomputation
```

Force refresh when source data is updated:

```python
CachingTransformer(DateParser("date"), store, "prices_msft_parsed", force=True).transform(raw)
```

## Notes

- `data/irp.duckdb` is gitignored (under `data/`)
- DuckDB tables are queryable directly with any DuckDB client or SQL tool
- Schema is stored as JSON in metadata and restored on load
