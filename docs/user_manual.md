# User Manual — Feature Engineering & Regime

A plain-language guide to two parts of the platform you can use to make investing
decisions, even if you are not a quant. It explains what every section and chart means,
and gives you step-by-step workflows.

If a word is unfamiliar, check the **Glossary** at the end first.

---

## Part 1 — The big picture

The platform turns raw company and market data into **signals** (numbers that hint whether
a stock will do well) and then helps you **decide** how to act on them. Two areas:

1. **Feature Engineering** (pages `/features` and `/feature-engineering`) — prepare a clean
   table of signals for a prediction model. Think of it as cooking: gather ingredients
   (features), clean them, and portion them into train/test sets so the model learns
   honestly.
2. **Regime** (page `/regime`) — read the overall "weather" of the market (calm vs stormy)
   and check which signals actually work in which weather, so you trust the right signal at
   the right time.

You do not have to use both. Feature Engineering feeds the prediction models; Regime is a
standalone decision aid. They complement each other.

---

## Part 2 — Feature Engineering

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

### 2.1 Why we clean and split (the one idea that matters)

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

### 2.2 Page `/features` — Dataset Builder

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

### 2.3 Page `/feature-engineering` — Feature Engineering

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

### 2.4 Feature-Engineering workflow for a decision

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

## Part 3 — The Regime page (`/regime`)

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

### 3.1 Section 1 — Dashboard

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

### 3.2 Section 2 — Regime-conditioned factors

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

### 3.3 Section 3 — Regime-gated backtest

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

### 3.4 Section 4 — Tactical allocation (cross-asset trend)

The question: **which asset class has the wind at its back right now?**

Click **Run tactical**. You get a ranked bar + table of trend strength across asset classes —
US equities, Nasdaq, Treasuries, gold, commodities, credit, crypto — averaged over 3/6/12
months.

**How to use it:** a relative-strength overview for allocation context. If Treasuries and gold
top the ranking while equities lag, that is a defensive tilt in the market — consistent with a
risk-off regime upstairs. It is a descriptive overlay, **not** a sized portfolio or a
buy/sell order.

### 3.5 Section 5 — Single-asset Markov regime ("hedge-fund method")

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

### 3.6 A Regime workflow for a decision

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
