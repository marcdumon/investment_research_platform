# User Manual — Today Dashboard, Feature Engineering & Regime

A plain-language guide to the parts of the platform you can use to make investing
decisions, even if you are not a quant. It explains what every section and chart means,
and gives you step-by-step workflows.

If a word is unfamiliar, check the **Glossary** at the end first.

---

## Part 1 — The big picture

The platform turns raw company and market data into **signals** (numbers that hint whether
a stock will do well) and then helps you **decide** how to act on them. Three areas:

1. **Today** (page `/today`) — your daily one-page briefing: what regime is the market in,
   which stocks rank highest for that regime right now, how your watchlist is doing, and
   what the prediction model currently recommends.
2. **Feature Engineering** (pages `/features` and `/feature-engineering`) — prepare a clean
   table of signals for a prediction model. Think of it as cooking: gather ingredients
   (features), clean them, and portion them into train/test sets so the model learns
   honestly.
3. **Regime** (page `/regime`) — read the overall "weather" of the market (calm vs stormy)
   and check which signals actually work in which weather, so you trust the right signal at
   the right time.

You do not have to use all three. `/today` is the daily starting point. Feature Engineering
feeds the prediction models. Regime is a deeper standalone decision aid. They complement
each other.

---

## Part 2 — The Today page (`/today`)

### 2.1 What it is

`/today` is your **daily decision surface** — a single page that answers three questions
the moment you open it:

1. **What is the market doing right now?** (the macro regime)
2. **Which stocks are best positioned for that regime?** (the ranked top names)
3. **How is my watchlist and my model doing?** (standing + model picks)

It does not tell you to buy or sell. It is a structured starting point that reduces the
daily "where do I even look?" problem to a few minutes of reading.

---

### 2.2 The controls

Before pressing **Run**, set these filters (all optional):

| Control | What it does |
|---|---|
| **Filing** | Annual (A) or Quarterly (Q) fundamental data for the factor scores. Annual is more stable; Quarterly reacts faster to earnings. |
| **Signal** | Which composite signal to rank stocks by. *Auto (regime)* picks automatically: momentum in risk-on, composite normally, quality in risk-off. You can override. |
| **Sector** | Restrict the ranked list to one sector. Leave blank for the whole universe. |
| **Watchlist** | Choose a saved watchlist to see how those names score (section 3). |
| **Min cap ($B)** | Minimum market cap. Default $1B removes micro-cap names whose extreme ratios distort rankings. |
| **Top N** | How many names to show in the ranked list. |

Press **Run** after changing any control.

---

### 2.3 Section 1 — Regime & playbook

This is the macro context. It reads the same cross-asset data as the full `/regime` page
but just shows the current state without charts.

**What each chip means:**

- **Regime** — `risk-on`, `neutral`, or `risk-off`. The colour tells you instantly: green
  is calm/rising, grey is uncertain, red is stress.
- **Risk score** — a number from 0 (most defensive) to 100 (most aggressive). Derived from
  yield-curve slope, equity trend, volatility, dollar, and commodities. A score above 60 is
  risk-on; below 40 is risk-off.
- **Playbook** — a plain-language sentence about what the regime implies: e.g. "Lean into
  trend; full gross" in risk-on, or "Defensive; tilt to quality, reduce gross" in risk-off.
  This is a fixed, hand-written rule — not a prediction.
- **Signal used** — which composite the top-names ranking actually used (may differ from
  your dropdown if you left it on Auto).
- **Regime as-of / Names as-of** — the dates the regime reading and the factor snapshot
  come from. If the names date is earlier than the regime date, the page shows a warning
  (normal near quarter-ends; will update when you next run Precompute on `/features`).

The link at the bottom opens the full `/regime` page for deeper analysis.

---

### 2.4 Section 2 — Top names

A table of the highest-ranked stocks for the active signal and your sector/market filters,
**from the full universe** (not your watchlist — that has its own section).

Columns:

- **Rank** — 1 = highest composite score.
- **Company Name / Sector** — identifiers.
- **mktcap** — market cap in $B. Helps you see how investable a name is.
- **score** — the composite signal value after rank-normalization. Higher = better relative
  positioning according to the chosen signal.
- **Factor columns** — the individual factor values that make up the composite (e.g. `pe`,
  `roe`, `mom_12_1`). These let you see *why* a name scored high.

**Clicking a row** opens that stock on the `/analysis` page while keeping `/today` intact.
This is the intended workflow: scan the list, click names of interest, review the price
history and fundamentals on `/analysis`, come back.

**Why not just sort by P/E?** The composite blends several factors so a stock must look
attractive on multiple dimensions at once, not just one. This reduces the chance of ranking
cheap-but-broken companies highly.

---

### 2.5 Section 3 — Watchlist standing

Shows the same composite score for each name on your chosen watchlist, plus:

- **percentile** — where each name stands in the full universe (0 = bottom, 100 = top).
  A name at the 80th percentile scores better than 80% of all stocks.
- **mom_12_1** — 12-month minus 1-month price momentum. Positive means the stock has been
  rising over the past year (after excluding the last month to avoid the short-term
  reversal effect).

Use this to quickly check: are your positions still well-positioned by the factor model,
or have they drifted to the bottom of the ranking?

This section only appears when you select a watchlist from the control bar.

---

### 2.6 Section 4 — Model picks

Shows the top names from the most recent **prediction export** — the output of a model
notebook (baseline linear or classifier). Unlike sections 2 and 3, which use hand-crafted
factor composites, this section shows what a trained machine-learning model predicts.

Columns depend on the model type:

- **pred** — predicted forward return (regression model). Higher = model expects more gain.
- **score** — expected quantile bucket (classifier model). Higher = model expects the stock
  to land in the best return bucket.
- **fwd_ret** — the actual forward return, if the prediction date is now in the past. Use
  this to see how the model performed.

The page shows the **source filename** and **prediction date** so you know how fresh the
model is.

**If this section says "No model export yet":** Run one of the model notebooks
(`baseline_linear_model.ipynb` or `baseline_classifier.ipynb`) and call
`save_predictions(res.predictions, 'baseline')` at the end. The file lands in
`data/model_predictions/` and `/today` picks it up automatically.

Clicking a row navigates to `/analysis` for that stock, same as section 2.

---

### 2.7 Today workflow — step by step

**Daily morning routine (5 minutes):**

1. Open `/today`, press **Run**.
2. Read the **Regime chip** (colour + score). Has it changed since yesterday?
3. Read the **Playbook chip** — one sentence about how aggressive to be.
4. Scan the **Top names table**. Any names you recognise? Any new entries near the top?
5. Click 1–3 names of interest → `/analysis` → check price chart and recent fundamentals.
6. Check **Watchlist standing** — are your holdings still near the top or have they slipped?
7. If a model is running, glance at **Model picks** — do they confirm or differ from the
   factor ranking?

**When regime changes** (e.g. flips from risk-on to risk-off):

- The playbook updates automatically.
- The Signal used will switch (e.g. momentum → quality).
- The top-names list will show different stocks. Names that scored well on momentum may
  drop; quality names rise.
- Consider reviewing current positions against the new ranking.

**When names date lags regime date:**  
A small yellow warning appears. This is normal near quarter-ends. The factor snapshot is
updated quarterly; the regime reads daily prices. The lag is at most ~3 months. Run
**Precompute** on the `/features` page (select the current year + variant, press Precompute)
to catch up.

---

## Part 3 — Feature Engineering (pages `/features` and `/feature-engineering`)

Goal: produce a clean dataset where each row is "one stock on one date" with a set of
signal columns and a **label** (what actually happened next), ready for a model to learn
from.

This is a **two-page workflow**:

- `/features` (**Dataset Builder**) — choose the universe, pick signals, attach the label,
  build and export the **raw** dataset.
- `/feature-engineering` (**Feature Engineering**) — load that dataset, look for problems,
  clean them, scale, split into train/validation/test, and export the final files.

> Why two pages? Building (what to include) and cleaning (how to condition it) are different
> jobs. Keeping them apart means you can rebuild ingredients without redoing the cleaning,
> and the cleaning is always done in one auditable place.

### 3.1 Why we clean and split (the one idea that matters)

A model is only useful if it works on data it has **never seen**. The cardinal sin is
**look-ahead / leakage**: letting the model peek at the future during training. It will look
brilliant in testing and then fail with real money.

Everything on the Feature-Engineering page exists to prevent leakage:

- **Splitting** keeps a slice of history aside (the **test** set) that the model never trains
  on, so you get an honest score.
- **Scaling** (rescaling numbers to a common range) is **fit on the training slice only**,
  then applied to the rest. Fitting it on everything would leak.
- **Cleaning** outliers is likewise fit on the training slice.

You do not have to understand the math. Just follow the order the page enforces: **clean →
handle missing → split → scale → export**. The page fits each step on the right slice for
you.

### 3.2 Page `/features` — Dataset Builder

**Row 1 — Universe & grid.** Choose *what stocks and how often*:

- **Start/End year** — the history range.
- **Grid frequency** — how often you sample each stock. `Q`/`A` (quarterly/annual) = one row
  per stock per filing; `M`/`W`/`D` (monthly/weekly/daily) = a price sequence per stock.
- **Variant `A`/`Q`** — use annual or quarterly fundamentals.
- **Market / Sector / Watchlist** — narrow the universe (e.g. only US tech, or your saved
  list).
- **Precompute** — warms the cache for the chosen years so building is fast. If a build says
  "run precompute_all first", click this once and wait.

**Row 2 — Add steps.** This is how you pick signals. Choose an operation, the column(s) it
uses, and parameters:

- **Base** — include a raw signal as-is (e.g. `pe`, `roe`, `mom_12_1`).
- **Lag / Diff / PctChange / Rolling** — time operations (value N periods ago, change since
  then, rolling average). Useful for "is this improving?".
- **Ratio / Product / Log / Winsorize** — math combinations of two signals.
- **Norm** — rank or z-score the signal across stocks on each date.
- **Quick pack** — add a whole family at once (e.g. all "value" signals). The fast way to
  start.
- **Batch add** — Column(s) is multi-select and the k/window boxes accept lists/ranges
  (`1,2,4` or `1-20`), so one click can add many variants.

Each step appears in a stack; delete any with its ✕.

**Row 3 — Label (the answer the model learns).**

- **Horizon** — how far ahead you measure the outcome (21/63/126/252 trading days ≈ 1
  month / quarter / half / year).
- **Mode** — `continuous` (predict the actual forward return), `binary` (up vs down), or
  `quantile` (which performance bucket). For a first pass, `continuous` is simplest.

**Row 4 — Run / Export / Recipes.**

- **Run Build** — builds the table and shows a 50-row preview + row/column/ticker counts.
- **Export parquet / CSV** — saves the raw dataset for the next page.
- **Save / Load recipe** — store this whole configuration under a name to reuse later.

**What you do here:** assemble the ingredients and the label, click Run Build, then either
Export or just move to `/feature-engineering` (which can pick up your last build in memory).

### 3.3 Page `/feature-engineering` — Feature Engineering

**Section 1 — Source.** Either **Use last build** (the dataset you just made on `/features`)
or load a previously exported file. The chosen dataset is what every step below operates on.

**Section 2 — Inspect & clean (the heart of the page).**

- **Run Inspect** builds a summary table of every signal column and automatically selects the
  worst-behaved one. "Worst" usually means **heavy-tailed** — a few extreme values that can
  wreck a model.
- The **column picker** (multi-select) drives a **histogram** per selected column — a picture
  of that signal's distribution. Tall bar in the middle with a few stragglers far out = heavy
  tail.
- **Action** decides how to tame a column:
  - **Clip (winsorize)** — pull the most extreme values in to a cut point (set by `p`, e.g.
    1%). The histogram shows dashed lines where the cut lands. **Side** lets you clip only the
    left, only the right, or both tails.
  - **Log** — compress big values with a signed-log transform. Good for quantities that span
    orders of magnitude.
  - **Drop** — remove the column entirely.
  - **None** — mark as reviewed/keep (no change), so it leaves the to-do list.
- **Apply to selected** records your choice for every selected column. Applied columns drop
  out of the picker; delete the stack entry to bring them back.
- **Exclude tickers** — remove specific stocks (case-insensitive). Click **Apply exclude** (or
  press Enter) and watch their rows leave the histogram. Use this for known-bad data.

You iterate: inspect → pick the worst column → choose an action → apply → re-inspect. Stop
when nothing looks pathological.

**Section 4 — Missing values, scale & split.**

- **Missing features** — `Drop rows` (remove rows with gaps) or `Impute median-per-date`
  (fill gaps with that date's median, leak-free). Rows with a missing label are always
  dropped. Every action is counted in the notes, so nothing disappears silently.
- **Scale** — `minmax` (squash to 0–1) or `robust` (median/IQR, less sensitive to outliers);
  **scope** = `date` (per cross-section, always safe), `global` (one scaler on the training
  years), or `ticker` (per stock). If unsure, `robust` + `date`.
- **Split** — `date` (chronological: oldest = train, newest = test — the realistic choice for
  investing) or `ticker` (whole stocks held out). Ratios default 0.7 / 0.15 / 0.15.

**Section 5 — Prepare & export.**

- **Run Prepare** — applies everything (exclude → clean → missing → split → scale) and shows a
  preview, counts, and notes explaining every change.
- **Export parquet / CSV** — writes the dataset. If you set a split, you get **three files**
  (`_train`, `_valid`, `_test`).
- **FE recipes** — save/load the whole cleaning+scaling+split configuration.

### 3.4 Feature-Engineering workflow for a decision

1. On `/features`: pick your universe, add a Quick pack (e.g. "value" + "quality"), set a
   label (continuous, 252d horizon), **Run Build**.
2. On `/feature-engineering`: **Use last build** → **Run Inspect**. Tame the few heavy-tailed
   columns (Clip at 1% both sides is a safe default). Exclude any obviously broken tickers.
3. Set **Missing = Impute**, **Scale = robust/date**, **Split = date**. **Run Prepare**, read
   the notes.
4. **Export** the three files.
5. Open a model notebook (`baseline_linear_model.ipynb` or `baseline_classifier.ipynb`), load
   the export, and read the output: a positive **mean IC** and an upward **quintile** chart
   mean your signals have real predictive power. A flat/negative result means they do not —
   better to know now than after trading them.

The decision you get: **which signals are worth trusting**, measured honestly on data the
model never trained on.

---

## Part 4 — The Regime page (`/regime`)

Goal: know the market's overall "weather", and use it to trust the right signals and time
your risk.

A **regime** is a market state. This page classifies it two ways:

- **Rule-based risk score (0–100)** — a transparent dial. High = "risk-on" (calm, trending up,
  healthy); low = "risk-off" (stress). You can see exactly which inputs moved it.
- **HMM states** — a statistical model that, on its own, finds a handful of recurring market
  states from the same inputs. A second opinion to the rule dial.

Inputs are **cross-asset**: the US yield curve, stock-market trend and volatility, the US
dollar, and commodities. These collectively describe the macro backdrop better than stocks
alone.

### 4.1 Section 1 — Dashboard

Controls: **Period** (how far back to show) and **HMM states** (how many market states to
find — 3 is a good default). Click **Run dashboard**.

What you see:

- **Chips** — the current **Rule regime** (risk-on / neutral / risk-off), the **Risk score**
  (0–100), the current **HMM state**, and **State persistence** (how likely the market stays
  in this state tomorrow — high means regimes are sticky).
- **^SPX with risk-off shaded** — the S&P 500 over time, with stress periods shaded red. Quick
  gut-check: do the red bands line up with crashes you remember (2008, 2020, 2022)? They
  should.
- **Rule risk score line** — the 0–100 dial over time, with the 40 and 60 thresholds marked.
- **Today's risk-on contribution by feature** — a bar per input showing what is pushing the
  score up (green) or down (red) *right now*. This is the "why" behind the call: e.g. "score
  is low mainly because volatility is high and the yield curve is inverted".
- **HMM transition matrix** — a small grid of probabilities of moving between states. Mostly
  for the curious; the diagonal (staying put) is usually high.

> Note: the HMM timeline is fitted on the whole history (in-sample) — it is for *seeing* the
> states, not for trading signals. The conditioning and gating below use only past data
> (no look-ahead).

**How to use it:** a one-glance read of the macro backdrop. Low score + red shading building
= be cautious. High score + green contributions = supportive backdrop.

### 4.2 Section 2 — Regime-conditioned factors

The key question: **does my favourite signal actually work in every market, or only some?**

Controls: **Signal** (a single factor like `pe`, or a ready-made **Composite** like "value"),
**Horizon**, **Rebalance**. Click **Run conditioning**.

What you see — two bar charts split by regime (risk-off / neutral / risk-on):

- **Mean IC by regime** — the signal's average predictive power in each regime. **IC**
  (information coefficient) is a correlation between the signal and what happened next: above
  ~0.03 is useful, higher is better, negative means it points the wrong way.
- **ICIR by regime** — the *consistency* of that power (higher = more reliable).

**How to use it:** suppose "value" shows IC 0.08 in neutral but ~0 in risk-on. Lesson: lean on
value in calm markets, do not expect it to save you in a euphoric run. You learn **when** each
signal earns its keep.

> If you see a warning about an empty IC series, the factor cache is cold for that period —
> click **Precompute** on `/features` for those years, then retry.

### 4.3 Section 3 — Regime-gated backtest

The question: **would timing this signal by regime have helped?**

Controls: **Signal**, **Allowed regimes** (the states in which you let the strategy trade —
e.g. only risk-on + neutral), **Horizon**, **Rebalance**. Click **Run gated**.

What you see:

- **Chips** — **Gated Sharpe** vs **Always-on Sharpe** (return per unit of risk; higher is
  better), and **max drawdown** for each (the worst peak-to-trough loss; smaller is better).
- **Cumulative L/S return** — two lines: the always-on strategy (grey) vs the regime-gated one
  (coloured). If the gated line ends higher and dips less, regime timing added value.

**How to use it:** if sitting out risk-off periods raises Sharpe and shrinks the drawdown, a
regime filter is worth adding to that strategy. If it barely changes, the signal is robust
across regimes and you can keep it always-on.

> Honest-test note: regimes are labelled using only information available at each date
> (no look-ahead), so the comparison is fair.

### 4.4 Section 4 — Tactical allocation (cross-asset trend)

The question: **which asset class has the wind at its back right now?**

Click **Run tactical**. You get a ranked bar + table of trend strength across asset classes —
US equities, Nasdaq, Treasuries, gold, commodities, credit, crypto — averaged over 3/6/12
months.

**How to use it:** a relative-strength overview for allocation context. If Treasuries and gold
top the ranking while equities lag, that is a defensive tilt in the market — consistent with a
risk-off regime upstairs. It is a descriptive overlay, **not** a sized portfolio or a
buy/sell order.

### 4.5 Section 5 — Single-asset Markov regime ("hedge-fund method")

This is the per-asset timing tool popularised in trading videos as the "Markov hedge-fund
method". It works on **one asset's own price** (a stock, crypto, ETF) rather than the macro
backdrop.

The idea in four steps:

1. **Label each day** bull / sideways / bear from the trailing return: if the last *N* days
   (your **Lookback**, default 20) returned more than the **Threshold** (default 5%) it is
   `bull`; less than −5% is `bear`; in between is `sideways`.
2. **Build a transition matrix** — a 3×3 grid of "if today is X, how often is tomorrow Y".
   The diagonal is **persistence** (how sticky each state is).
3. **Project forward** — multiplying the matrix by itself gives the odds *k* days out; far
   enough out it settles to the **stationary distribution** (the long-run mix).
4. **Make a signal** — `P(bull tomorrow) − P(bear tomorrow)`. Positive ⇒ lean long, negative
   ⇒ lean short; the size of the number is your conviction.

Controls: **Instrument**, **Lookback**, **Threshold %**, **Sampling**, **HMM overlay**,
**Backtest horizon**, then **Run Markov**.

What you see:

- **Chips** — current state, the **signal** value and its direction, **persistence**
  (stickiness of the current state), **HMM agreement** (how often the threshold-free HMM
  states agree with the rule states — a confidence check), and which **sampling** you used.
- **Transition matrix** heatmap and **state forecast** (probability of each state vs days
  ahead, with the long-run/stationary level dashed).
- **Walk-forward backtest** — the honest test: builds the matrix from *past data only* at each
  step, trades the signal, and plots it against simply buying and holding, with both Sharpe
  ratios.

Two things to take seriously:

- **Sampling matters.** "Overlapping" (the video's default) samples every day, but two
  adjacent 20-day windows share 19 days, which makes states look far stickier than they really
  are. **Non-overlapping** (the default here) samples every 20 days for an honest matrix. Flip
  between them and watch the persistence number drop — that drop is the illusion leaving.
- **Check the backtest before believing the signal.** Very often the strategy's Sharpe does
  **not** beat buy-and-hold (e.g. for big trending stocks). That means the matrix is not adding
  tradeable edge for that asset, however clever it looks. This is the reality check the hype
  videos skip.

**How to use it:** a quick, transparent read on one asset's momentum/mean-reversion character
and a built-in honesty check on whether timing it actually pays. Treat the signal as a lean,
not gospel, and always glance at the walk-forward result first.

### 4.6 A Regime workflow for a decision

1. **Run dashboard** (10Y). Read the current regime and the contribution bars — know the
   backdrop and why.
2. **Run conditioning** on the signal/composite you intend to use. Confirm it actually works
   in *today's* regime, not just on average.
3. **Run gated** on that signal to see whether avoiding bad regimes would have improved
   risk-adjusted return.
4. **Run tactical** for the cross-asset tilt as a sanity check.

The decision you get: **how much risk to take now, and which of your signals to trust in the
current market** — instead of trading one signal blindly through every kind of market.

---

## Glossary

- **Regime** — the market's overall state (risk-on / neutral / risk-off). Determined by
  cross-asset data: yield curve, equity trend, volatility, dollar, commodities.
- **Risk score** — 0 to 100; above 60 = risk-on, below 40 = risk-off. Derived from a
  weighted blend of expanding z-scored macro features; no fitting, fully transparent.
- **Playbook** — the hand-written rule mapping each regime to a recommended signal and
  gross-exposure stance. Changes automatically when the regime changes.
- **Composite** — several factors blended into one signal.
- **Percentile** — where a stock ranks in the universe; 80th percentile = better than 80%
  of all stocks on the chosen signal.
- **Cross-section** — all stocks scored on a single date.
- **Feature / signal** — a number describing a stock (e.g. P/E ratio, 12-month momentum).
- **Label** — what actually happened next (the answer the model learns), e.g. the forward
  return.
- **Factor** — a signal believed to predict returns (value, quality, momentum, ...).
- **Composite** — several factors blended into one signal.
- **Cross-section** — all stocks on a single date.
- **Look-ahead / leakage** — accidentally using future information; makes tests look great and
  live results terrible. The thing all the cleaning/splitting prevents.
- **Train / validation / test split** — slices of history: learn on train, tune on validation,
  judge honestly on the untouched test set.
- **Scaling** — rescaling numbers to a comparable range so no single signal dominates by units.
- **Winsorize / clip** — pull extreme outliers in to a cut point.
- **Heavy-tailed** — a signal with rare but huge values that distort models.
- **IC (information coefficient)** — correlation between a signal and the next return; the core
  measure of predictive power (>~0.03 useful).
- **ICIR** — IC divided by its variability; the *consistency* of the power.
- **Quintile chart** — sort stocks into five buckets by signal; a rising staircase means the
  signal ranks winners above losers.
- **Sharpe ratio** — return per unit of risk; higher is better.
- **Max drawdown** — the worst peak-to-trough loss; smaller is better.
- **Long-short (L/S)** — buy the top-ranked stocks, short the bottom; isolates the signal's
  edge.
- **Regime** — the market's overall state (risk-on / neutral / risk-off).
- **Risk-on / risk-off** — appetite for risk: on = calm and rising, off = stress and falling.
- **HMM (Hidden Markov Model)** — a statistical method that discovers recurring market states
  on its own.
- **Transition matrix** — a grid of probabilities of moving from today's state to tomorrow's.
- **Persistence / stickiness** — how likely a state is to repeat tomorrow (the matrix diagonal).
- **Stationary distribution** — the long-run mix of states the matrix settles into far ahead.
- **Markov property** — the assumption that tomorrow depends only on today's state, not the
  full past.
- **Overlapping vs non-overlapping windows** — sampling every day (windows share data, inflates
  stickiness) vs every N days (honest). Prefer non-overlapping for the transition matrix.
- **Yield curve** — the gap between long and short government-bond rates; an inverted (negative)
  curve has historically preceded recessions.
- **Z-score** — how many standard deviations a value is from its average; a way to compare
  different signals on one scale.
