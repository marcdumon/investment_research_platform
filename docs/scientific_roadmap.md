# Scientific Next Steps — Investment Research Platform

## Context

Platform has solid data pipeline (68K fundamental rows, 45.5M price rows, 48 computed metrics, screener, PCA) but zero empirical validation. Screener picks stocks with no evidence they outperform. Five steps below build a scientifically rigorous validation chain; each step depends on the previous.

---

## Step 1 — Forward Returns Module (Small)

**New files:** `src/irp/features/returns.py` + `notebooks/returns_analysis.ipynb`

Functions in `returns.py`:
- `forward_returns(prices, horizons=(21, 63, 126, 252))` — log returns per ticker/date at each horizon using `merge_asof`. Anchor: `date + horizon days`.
- `benchmark_returns(prices, ticker="SPY", horizons=...)` — same for SPY (confirmed in DB, 5,329 rows back to 2005).
- `excess_returns(forward_df, benchmark_df)` — subtract benchmark return at same horizon/date.
- `risk_adjusted(prices, rf_rate=0.04)` — rolling 252-day Sharpe, Sortino, Calmar per ticker.

Notebook: histogram of 1Y forward return distribution across universe (Plotly).

**Key guard:** `prices.date` is VARCHAR — cast to datetime before rolling ops (pattern from `_momentum()` in `metrics.py`).

**Unlocks:** All steps below need `forward_returns` as the outcome variable `y`.

---

## Step 2 — Signal Significance Testing (Medium)

**New files:** `src/irp/features/signals.py` + `notebooks/metric_signal_test.ipynb`

Functions in `signals.py`:
- `metric_return_panel(metrics_df, forward_df)` — join `compute_metrics(latest=False)` to `forward_returns` using `merge_asof` on `Publish Date`. **Never `Report Date`** — that pre-dates the filing becoming public.
- `rank_ic(panel_df, metric_cols, horizon)` — cross-sectional Spearman IC per rebalance date per metric.
- `ic_summary(ic_series)` — mean IC, IC std, IR, t-stat, p-value (`scipy.stats.ttest_1samp`), fraction positive.

Notebook: bar chart of mean IC ± 95% CI ranked by |IC|; table with t-stats, Bonferroni-corrected p-values (48 tests → threshold 0.001).

**Unlocks:** Only test metrics that pass significance in step 3. Prevents 2-3 spurious positives expected at p<0.05 from 48-way comparison.

---

## Step 3 — Backtesting Engine (Medium)

**New files:** `src/irp/features/backtest.py` + `notebooks/backtest.ipynb`

Functions in `backtest.py`:
- `build_signal_universe(metrics_panel, signal_col, date_col="Publish Date")` — rank tickers by signal at each rebalance date; return top-N and bottom-N decile lists.
- `run_backtest(prices, signal_universe, hold_days=63, top_n=20, equal_weight=True)` — equal-weight portfolio, hold for `hold_days`, compute gross return, benchmark return, excess return, max drawdown per period.
- `portfolio_stats(trade_pnl_df, rf=0.04)` — Sharpe, Sortino, Calmar, win rate, annualized alpha vs SPY.

Notebook: equity curve (log-scale), rolling Sharpe, drawdown chart, trade return histogram — all Plotly.

**Survivorship bias check:** Run backtest twice (active-only vs. including `_delisted` tickers); report delta. This quantifies artifact alpha.

**Unlocks:** Step 4 needs a validated baseline equity curve to compare against optimized weights.

---

## Step 4 — Portfolio Optimization (Medium)

**New file:** `src/irp/features/portfolio.py` (add second equity curve to step-3 notebook)

Functions:
- `covariance_ledoit_wolf(returns_panel)` — shrunk cov via `sklearn.covariance.LedoitWolf` (avoids singular matrix with 20 stocks, 252 days).
- `mean_variance_weights(expected_returns, cov_matrix, risk_aversion=1.0, max_weight=0.10)` — `scipy.optimize.minimize` SLSQP.
- `max_sharpe_weights(expected_returns, cov_matrix, rf=0.04, max_weight=0.10)` — maximize Sharpe under weight constraints.

Notebook addition: overlay Sharpe-optimized curve on equal-weight baseline from step 3.

**No new packages needed:** scipy + sklearn already installed.

---

## Step 5 — Factor Decomposition (Large)

**New files:** `src/irp/features/factors.py` + `notebooks/factor_model.ipynb`

Functions:
- `compute_factor_returns(prices, factor_tickers={"market":"SPY","size":"IWM","value":"IVE","momentum":"MTUM"})` — use ETF proxies from existing price data. Fallback: construct long-short portfolios from IC-ranked metrics (step 2).
- `factor_exposures(stock_returns, factor_returns, lookback_days=252)` — rolling OLS per ticker via `statsmodels.OLS`.
- `alpha_decomposition(metrics_df, factor_exposures_df)` — merge exposures to cross-section.

Notebook: factor exposure heatmap by sector; scatter of raw vs. beta-adjusted return; bar of factor attribution (how much of step-3 return is beta vs. genuine alpha).

**Why last:** Validates whether screener alpha is real or just value/momentum beta that's widely arbitraged.

---

## Dependency Chain

```
Step 1: returns.py     → forward_returns panel (y variable)
    ↓
Step 2: signals.py     → which of 48 metrics are significant
    ↓
Step 3: backtest.py    → does top-IC metric actually outperform?
    ↓
Step 4: portfolio.py   → does risk-aware sizing improve Sharpe?
    ↓
Step 5: factors.py     → is the alpha real or just beta?
```

---

## Critical Files to Read Before Implementing

- `src/irp/features/metrics.py` — `_momentum()`, `_attach_price_asof()`, `_build_base()`, `compute_metrics(latest=False)`
- `src/irp/features/base.py` — Feature base class
- `notebooks/screener.ipynb` — boilerplate: DuckDB connect, `_ROOT` path, dark-theme detection, Plotly conventions
- `notebooks/pca_fundamentals.ipynb` — pattern for full-panel feature computation
- `src/irp/store.py` — read-only load pattern

---

## Verification

Each step: run notebook end-to-end, confirm no errors, visually check output charts. Step 3 specific: equity curve must start at 1.0, end date within last 3 months of latest price data, SPY line must match known SPY performance. Step 5: market beta for SPY itself must be ~1.0 (sanity check).
