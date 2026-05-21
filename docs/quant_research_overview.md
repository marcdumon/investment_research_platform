# Quant Research Overview

_2026-05-21 (updated)_

---

## Data inventory

### Asset classes

| Asset class | Price data | Fundamentals |
|---|---|---|
| US stocks | Stooq OHLCV (45.5M rows, 1789–present) | SimFin income / balance / cashflow |
| DE stocks | Stooq OHLCV | SimFin income / balance / cashflow |
| Indices | Stooq OHLCV | None |
| Bonds | Stooq OHLCV | None |
| Currencies | Stooq OHLCV | None |
| ETFs | Stooq OHLCV | None |

Fundamentals gate what is analytically interesting. US + DE stocks are where the real work happens.

---

## Analysis categories

### 1. Fundamental ratios (cross-sectional snapshot)

Compute per ticker per period, joining latest prices:

| Ratio | Inputs | What it measures |
|---|---|---|
| P/E | Price / EPS | Cheapness vs earnings |
| P/B | Price × Shares / Equity | Cheapness vs book |
| EV/EBITDA | (Mkt cap + Debt − Cash) / EBITDA | Enterprise cheapness |
| P/FCF | Price / (CFO − Capex) | Cheapness vs free cash |
| Gross margin | Gross Profit / Revenue | Pricing power |
| Operating margin | Op Income / Revenue | Efficiency |
| Net margin | Net Income / Revenue | Bottom-line conversion |
| ROE | Net Income / Equity | Return on book |
| ROIC | EBIT(1−t) / (Debt + Equity − Cash) | Capital allocation quality |
| Net Debt/EBITDA | (Debt − Cash) / EBITDA | Leverage |
| Interest coverage | EBIT / Interest Expense | Solvency buffer |
| Current ratio | Current Assets / Current Liabilities | Liquidity |

**PIT alignment**: `irp.factors` uses SimFin's `Report Date` (fiscal period end) as the PIT cutoff. This is conservative but introduces up to ~30–60 day lookahead bias: results are typically published 4–8 weeks after the fiscal period ends. The stricter approach — use SimFin's `Publish Date` (when the filing was actually made public) — would eliminate this. Relevant for backtesting; less critical for exploratory cross-sections.

---

### 2. Growth metrics (time series per ticker)

Derived from quarterly / annual panels:

- Revenue YoY growth, 3Y CAGR
- EPS YoY growth, 3Y CAGR
- Gross margin expansion / contraction trend
- FCF conversion trend (FCF / Net Income)
- Accruals ratio: (Net Income − CFO) / Assets — high accruals predict underperformance (Sloan 1996)

**Noise problem**: SimFin fundamentals have known violations (accounting identity rules flag a meaningful share of rows). Filter or winsorize before computing ratios. Division by near-zero causes explosion (EV/EBITDA when EBITDA < 0, P/E for loss-making firms).

---

### 3. Price features (all 14k instruments)

**Use log returns throughout**: `log_ret = ln(P_t / P_{t-1})`

Log returns are time-additive (`Σ daily log returns = horizon log return`), approximately normally distributed, symmetric, and correct for volatility computation. Simple returns compound and do not add — they are only needed for cross-sectional portfolio P&L attribution, where portfolio return = `Σ w_i × simple_return_i`.

**Base fact**: store daily log return once. Derive any horizon by summing — no need to precompute multiple return windows separately.

| Feature | Formula | Notes |
|---|---|---|
| `log_ret_1d` | `ln(C / C_lag1)` | Base fact |
| `log_ret_1m` | `ln(C / C_lag21)` | Sum of 21 daily log returns |
| `log_ret_6m` | `ln(C / C_lag126)` | |
| `log_ret_12m` | `ln(C / C_lag252)` | |
| `mom_12_1m` | `ln(C_lag21 / C_lag252)` | Classic Jegadeesh-Titman. Skip last month to avoid short-term reversal. |
| `mom_6_1m` | `ln(C_lag21 / C_lag126)` | Shorter window, higher turnover |
| `reversal_1m` | `ln(C / C_lag21)` | Short-term mean reversion signal |
| `vol_21d` | `std(log_ret_1d, 21d) × √252` | Annualised realised vol |
| `vol_63d` | `std(log_ret_1d, 63d) × √252` | |
| `high_52w_ratio` | `C / max(H, 252d)` | Proximity to 52wk high; stocks near high tend to outperform |
| `ma200_ratio` | `C / SMA(C, 200d)` | Trend filter |
| `ma50_ratio` | `C / SMA(C, 50d)` | |

**Stooq price caveat**: NOT dividend-adjusted. For momentum signals this matters less (relative ranking is stable). For absolute return analysis, dividend payers are systematically understated. Yahoo adjusted prices solve this but have shorter history.

---

### 4. Composite factor scores

Factors work better combined. Classic combinations:

**Value + Momentum (Fama-French-Carhart)**
Cheap stocks (low P/B or P/E) AND positive 6–12m price momentum. Avoids value traps (cheap but still falling).

**Quality + Momentum**
High ROE, high gross margin, low accruals AND positive momentum. Tends to be most stable combination in practice.

**Piotroski F-Score** (9-point binary checklist from accounting data)

| Category | Signal |
|---|---|
| Profitability | Positive ROA, positive CFO, rising ROA, CFO > Net Income |
| Leverage / dilution | Falling debt ratio, rising current ratio, no new share issuance |
| Efficiency | Rising gross margin, rising asset turnover |

Score 0–9. High F-score among cheap stocks = historically strong outperformer. Entirely computable from `balance`, `income`, `cashflow` tables.

**Magic Formula (Greenblatt)**
Rank by earnings yield (EBIT/EV) + rank by ROIC → add ranks → sort. Simple, replicable, well-documented historical returns.

---

### 5. Cross-asset / macro signals

With bonds + currencies + indices:

- **Yield curve slope**: long bond price vs short T-bill → proxy for 10Y−2Y spread → regime signal
- **Risk-on/risk-off**: equity index momentum vs bond index momentum
- **FX carry**: high-yielding vs low-yielding currency pairs (crude proxy from price history)
- **Equity performance conditional on regime**: does momentum work better in rising yield environments? Does value recover when yield curve steepens?

---

## Key methodological issues

**1. Survivorship bias**
Universe includes current tickers. Delisted companies (failures) likely missing or sparse in fundamentals. Inflates backtested returns if uncorrected. Mitigant: use catalog coverage to filter to tickers with continuous data over the test window.

**2. Point-in-time joins**
`irp.factors` uses `Report Date` (fiscal period end) as the PIT cutoff — introduces ~30–60 day lookahead bias. To eliminate it, switch to SimFin's `Publish Date` in `pit_latest` / `pit_ttm`.

**3. Ratio explosion**
When EBITDA, earnings, or book value is negative / near-zero, ratios blow up. Fix: winsorize at 1st/99th percentile, or exclude negatives from valuation sorts (but still include in growth/quality sorts).

**4. Stooq vs Yahoo prices**
- Return-based signals (momentum, total return): prefer Yahoo adjusted prices.
- Valuation (market cap computation): Stooq is fine; dividends don't affect market cap.

**5. Currency heterogeneity**
German stocks report in EUR. Cross-country factor portfolios need USD-EUR conversion. Manageable if sticking to US-only universe initially.

**6. SimFin data noise**
Filter `violations == 0` rows before ratio computation, or flag affected tickers. Accounting identity violations affect a non-trivial share of rows.

---

## Suggested experiment sequence

| Phase | Experiment | Status |
|---|---|---|
| 1 | Cross-sectional snapshot — compute all ratios for latest period, explore distribution by sector | **Done** |
| 2 | Momentum factors — 12-1m, 6-1m log returns, realised vol, MA200 ratio | **Done** |
| 3 | Factor backtest — Spearman IC series + quintile cumulative returns for any factor | **Done** |
| 4 | Piotroski F-Score — 9 accounting signals (profitability, leverage, efficiency), score 0–9 | Pending |
| 5 | Factor combination — Value + Quality + Momentum composite score | Pending |

**Phase 1** (`irp.factors`): 15 factors across valuation + profitability. PIT-safe via `pit_latest` / `pit_ttm`. Quarterly variant uses TTM aggregation. UI: `/factors` (cross-section ranking + per-ticker factor history).

**Phase 2** (`irp.factors.momentum`): 4 price-based factors — `mom_12_1`, `mom_6_1`, `vol_21d`, `ma200_ratio`. Calendar-day lags (30/182/365 days). Skips last month to avoid short-term reversal bias. Integrated into `cross_section()` and `ticker_factor_history()`.

**Phase 3** (`irp.factors.backtest`): `run_backtest(factor, horizon_days, start_date, end_date)` fetches raw data once, iterates over quarterly (or annual) rebalance dates in memory, computes Spearman IC per date and equal-weight quintile cumulative log returns. Exposed in the Dash UI at `/backtest`.

---

## Realistic expectations

| Factor | Historical evidence | Post-2010 reality |
|---|---|---|
| Momentum | Strong, consistent | Still works, but crowded |
| Value (P/E, P/B) | Strong pre-2007 | Weak; recovering post-2022 |
| Quality (margins, ROE) | Solid | Expensive but defensive |
| Piotroski | Strong among small caps | Harder in large caps |
| Low volatility | Strong risk-adjusted | Crowded, compressed alpha |
| Combined | Most robust | Still the best approach |

**Main caveat**: SimFin covers ~6.6k companies, mostly large/mid cap US. This reduces the small-cap effect, which is where many academic results are strongest. Expect more muted factor returns than papers report.

---

## Recommended starting points

Piotroski F-Score and momentum are the cleanest first experiments given data quality constraints. Value requires the most careful point-in-time work and ratio computation hygiene.
