# Transformation Layer

Layer C of the research platform. Transforms take a `Dataset` and return a new `Dataset` — no side effects, no hidden state.

All transforms implement:

```python
class Transformer(ABC):
    def transform(self, dataset: Dataset) -> Dataset: ...
```

---

## Data sources

| Source | File / API | Key columns | Date dimension |
|--------|-----------|-------------|----------------|
| Equities | `d_us_txt.zip` | date, open, high, low, close, volume | daily |
| Currencies (FX) | `d_world_txt.zip` | date, open, high, low, close, volume | daily |
| Indices | `d_world_txt.zip` | date, open, high, low, close, volume | daily |
| Fundamentals | SimFin income / balance / cashflow | Ticker, Publish Date, financials | quarterly |
| Companies | SimFin companies | Ticker, Company Name, IndustryId, ... | static |
| Industries | SimFin industries | IndustryId, Industry, Sector | static |

---

## Transforms

### 1. DateParser

Convert a string date column to `datetime64[ns]`.

```python
DateParser(col: str = "date")
```

| | |
|---|---|
| **Input** | Dataset with a string date column |
| **Output** | Same dataset, column cast to `datetime64[ns]` |
| **Why** | Stooq returns dates as `"YYYY-MM-DD"` strings. All downstream transforms require sortable, arithmetic-capable dates. |

---

### 2. Cleaner

Remove structural noise from a dataset.

```python
Cleaner(drop_all_nan: bool = True, clip_percentile: float | None = None)
```

| | |
|---|---|
| **Input** | Any dataset |
| **Output** | Same schema, fewer or modified rows |
| **Operations** | Replace `±inf` → `NaN`; drop rows where all numeric columns are `NaN`; optionally winsorise numeric columns at `[p, 1−p]` percentile |

---

### 3. ForwardFiller

Carry the last known value forward per entity across a date grid.

```python
ForwardFiller(
    date_col: str = "date",
    ticker_col: str = "Ticker",
    target_dates: pd.DatetimeIndex | None = None,
    limit: int | None = None,
)
```

| | |
|---|---|
| **Input** | Dataset with `date` (datetime64) and `ticker` columns, one row per (ticker, date) |
| **Output** | Dataset reindexed to `target_dates`, values filled forward per ticker |
| **Why** | Quarterly fundamentals reported once per quarter. Forward fill makes them available on every subsequent trading day until the next report. |
| **Algorithm** | Pivot wide (index=date, columns=ticker); reindex to target dates; `ffill(limit=limit)`; stack back to long |

> **Point-in-time correctness:** Pass `Publish Date` as `date_col`, not `Report Date`. The transform is unaware of this distinction — the caller is responsible.

---

### 4. DateAligner

Reindex a dataset to an explicit set of dates.

```python
DateAligner(
    target_dates: pd.DatetimeIndex,
    date_col: str = "date",
    ticker_col: str | None = "ticker",
    method: Literal["ffill", "bfill", "nearest", "none"] = "ffill",
)
```

| | |
|---|---|
| **Input** | Dataset with date column |
| **Output** | Dataset reindexed to `target_dates` — rows added or removed |
| **Why** | After forward filling fundamentals, snap them to the exact trading days from price data before joining. |

---

### 5. Joiner

Merge two datasets on shared key columns.

```python
Joiner(
    right: Dataset,
    on: list[str] = ["ticker", "date"],
    how: Literal["left", "inner", "outer"] = "left",
    suffixes: tuple[str, str] = ("_left", "_right"),
)
```

| | |
|---|---|
| **Input** | `left` Dataset passed to `transform()`; `right` injected at construction |
| **Output** | Merged dataset. `name = f"{left.name}+{right.name}"`, `source = "join"` |
| **Why** | `right` is fixed at construction so `Joiner` remains a valid `Transformer` (one-argument `transform`). |
| **Static joins** | For company/industry metadata (no date dimension), pass `on=["Ticker"]` or `on=["IndustryId"]` |

---

## Pipeline examples

### A — Equities + fundamentals (point-in-time)

```python
prices = DateParser("date").transform(StooqBulkSource("msft.us").fetch())
prices = Cleaner().transform(prices)

fund = DateParser("Publish Date").transform(SimFinFundamentalsSource("income").fetch())
fund = ForwardFiller(target_dates=prices.data["date"]).transform(fund)
fund = DateAligner(target_dates=prices.data["date"]).transform(fund)

joined = Joiner(right=fund, on=["Ticker", "date"]).transform(prices)
```

### B — Equity prices with sector / industry metadata

```python
companies  = SimFinFundamentalsSource("companies").fetch()
industries = SimFinFundamentalsSource("industries").fetch()

co_enriched     = Joiner(right=industries, on=["IndustryId"], how="left").transform(companies)
prices_enriched = Joiner(right=co_enriched, on=["Ticker"],    how="left").transform(prices)
```

### C — FX-adjusted prices

```python
eurusd         = Cleaner().transform(DateParser("date").transform(StooqBulkSource("eurusd.fx").fetch()))
eurusd_aligned = DateAligner(target_dates=prices.data["date"]).transform(eurusd)
prices_with_fx = Joiner(right=eurusd_aligned, on=["date"], suffixes=("", "_fx")).transform(prices)
# FX-adjusted close computed in Feature layer: close * close_fx
```

### D — Index benchmark

```python
spx         = Cleaner().transform(DateParser("date").transform(StooqBulkSource("^spx").fetch()))
spx_aligned = DateAligner(target_dates=prices.data["date"]).transform(spx)
# Feature layer: excess_return = equity_return - index_return
```
