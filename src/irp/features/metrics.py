"""Stock metrics: valuation, profitability, growth, health, cash flow, dividend, momentum, scores.

Reusable across screener, AI features, charts. Top-level entry: `compute_metrics`.
"""
from typing import Literal

import numpy as np
import pandas as pd

# --- period helpers ---

def _period_key(p: str) -> tuple[int, int]:
    yr = int(p[:4])
    if p.endswith("FY"):
        return (yr, 5)
    return (yr, int(p[-1]))


def _period_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_pk"] = df["period"].map(_period_key)
    df = df.sort_values(["Ticker", "_pk"]).drop(columns=["_pk"])
    return df


# --- base builders ---

def _filter_variant(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    return df[df["variant"] == variant].copy()


def _ttm_aggregate(
    df: pd.DataFrame,
    sum_cols: list[str],
    last_cols: list[str],
) -> pd.DataFrame:
    """Roll trailing 4 quarters per ticker.

    sum_cols: flow items to sum (Revenue, Net Income, OpCF, etc.)
    last_cols: stock items to take latest (Total Assets, Equity, Shares, etc.)
    Other identifying columns kept from latest row.
    """
    q = _filter_variant(df, "Q")
    q = _period_sort(q)

    sum_cols = [c for c in sum_cols if c in q.columns]
    last_cols = [c for c in last_cols if c in q.columns]

    grp = q.groupby("Ticker", group_keys=False)

    rolled_sum = grp[sum_cols].apply(lambda g: g.rolling(4, min_periods=4).sum())
    rolled_last = q[last_cols]

    # Identifiers: take from current row
    id_cols = ["Ticker", "period", "Publish Date", "Report Date"]
    id_cols = [c for c in id_cols if c in q.columns]

    out = pd.concat(
        [q[id_cols].reset_index(drop=True),
         rolled_sum.reset_index(drop=True),
         rolled_last.reset_index(drop=True)],
        axis=1,
    )
    # Need full 4 quarters; drop first 3 per ticker
    out = out.dropna(subset=sum_cols, how="any") if sum_cols else out
    return out


# Column groups per statement
_INCOME_FLOW = [
    "Revenue", "Cost of Revenue", "Gross Profit",
    "Operating Expenses", "Selling, General & Administrative", "Research & Development",
    "Depreciation & Amortization", "Operating Income (Loss)",
    "Interest Expense, Net", "Pretax Income (Loss)",
    "Income Tax (Expense) Benefit, Net",
    "Net Income", "Net Income (Common)",
]
_INCOME_LAST = ["Shares (Basic)", "Shares (Diluted)"]

_BALANCE_LAST = [
    "Cash, Cash Equivalents & Short Term Investments",
    "Inventories", "Total Current Assets",
    "Property, Plant & Equipment, Net",
    "Total Assets",
    "Short Term Debt", "Total Current Liabilities",
    "Long Term Debt", "Total Liabilities",
    "Retained Earnings", "Total Equity",
    "Shares (Basic)", "Shares (Diluted)",
]

_CASHFLOW_FLOW = [
    "Depreciation & Amortization",
    "Net Cash from Operating Activities",
    "Change in Fixed Assets & Intangibles",
    "Net Cash from Investing Activities",
    "Dividends Paid",
    "Net Cash from Financing Activities",
]


def _build_base(
    fundamentals: dict[str, pd.DataFrame],
    period: Literal["annual", "ttm"],
) -> pd.DataFrame:
    income = fundamentals["income"]
    balance = fundamentals["balance"]
    cashflow = fundamentals["cashflow"]

    if period == "annual":
        inc = _filter_variant(income, "A")
        bal = _filter_variant(balance, "A")
        cf  = _filter_variant(cashflow, "A")
    else:  # ttm
        inc = _ttm_aggregate(income, _INCOME_FLOW, _INCOME_LAST)
        bal = _filter_variant(balance, "Q")  # balance is point-in-time
        cf  = _ttm_aggregate(cashflow, _CASHFLOW_FLOW, [])

    # Select columns to merge
    inc_cols = ["Ticker", "period", "Publish Date"] + [c for c in _INCOME_FLOW + _INCOME_LAST if c in inc.columns]
    bal_cols = ["Ticker", "period"] + [c for c in _BALANCE_LAST if c in bal.columns]
    cf_cols  = ["Ticker", "period"] + [c for c in _CASHFLOW_FLOW if c in cf.columns]

    inc = inc[inc_cols].drop_duplicates(["Ticker", "period"])
    bal = bal[bal_cols].drop_duplicates(["Ticker", "period"])
    cf  = cf[cf_cols].drop_duplicates(["Ticker", "period"])

    base = inc.merge(bal, on=["Ticker", "period"], suffixes=("", "_bal"))
    base = base.merge(cf, on=["Ticker", "period"], suffixes=("", "_cf"))
    base["Publish Date"] = pd.to_datetime(base["Publish Date"])
    return base


# --- price helpers ---

def _latest_price(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.sort_values("date").groupby("ticker", as_index=False).tail(1)
    return p[["ticker", "date", "close"]].rename(columns={"date": "price_date"})


def _price_asof(base: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """For each (Ticker, Publish Date), find closest price on or before."""
    left = base[["Ticker", "Publish Date"]].sort_values("Publish Date").copy()
    right = (
        prices.assign(date=pd.to_datetime(prices["date"]))
        .sort_values("date")[["ticker", "date", "close"]]
        .rename(columns={"ticker": "Ticker", "date": "Publish Date"})
    )
    merged = pd.merge_asof(
        left, right,
        on="Publish Date", by="Ticker",
        direction="backward",
    )
    return merged.rename(columns={"close": "price"})


def _attach_price_asof(base: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach as-of-publish price + market_cap to every panel row."""
    px = _price_asof(base, prices)
    px = px.drop_duplicates(["Ticker", "Publish Date"])
    out = base.merge(px[["Ticker", "Publish Date", "price"]], on=["Ticker", "Publish Date"], how="left")
    out["market_cap"] = out["price"] * out["Shares (Diluted)"]
    return out


def _overlay_latest_price(b: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Replace `price` and `market_cap` with today's values (per ticker)."""
    lp = _latest_price(prices).set_index("ticker")["close"]
    out = b.copy()
    out["price"] = out["Ticker"].map(lp)
    out["market_cap"] = out["price"] * out["Shares (Diluted)"]
    return out


# --- safe division ---

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.where(b.notna() & (b != 0)).div(b.replace(0, np.nan))


# --- category helpers ---

def _valuation(b: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=b.index)
    mc = b["market_cap"]
    out["P/E"] = _safe_div(mc, b["Net Income"])
    out["Earnings Yield"] = _safe_div(b["Net Income"], mc)
    out["P/B"] = _safe_div(mc, b["Total Equity"])
    out["P/S"] = _safe_div(mc, b["Revenue"])
    fcf = b["Net Cash from Operating Activities"] + b.get("Change in Fixed Assets & Intangibles", 0)
    out["P/FCF"] = _safe_div(mc, fcf)
    debt = b["Short Term Debt"].fillna(0) + b["Long Term Debt"].fillna(0)
    cash = b["Cash, Cash Equivalents & Short Term Investments"].fillna(0)
    ev = mc + debt - cash
    ebitda = b["Operating Income (Loss)"] + b.get("Depreciation & Amortization_cf", b.get("Depreciation & Amortization", 0))
    out["EV/EBITDA"] = _safe_div(ev, ebitda)
    out["EV/Sales"] = _safe_div(ev, b["Revenue"])
    return out


def _profitability(b: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=b.index)
    out["Gross Margin"] = _safe_div(b["Gross Profit"], b["Revenue"])
    out["Operating Margin"] = _safe_div(b["Operating Income (Loss)"], b["Revenue"])
    out["Net Margin"] = _safe_div(b["Net Income"], b["Revenue"])
    out["ROE"] = _safe_div(b["Net Income"], b["Total Equity"])
    out["ROA"] = _safe_div(b["Net Income"], b["Total Assets"])
    debt = b["Short Term Debt"].fillna(0) + b["Long Term Debt"].fillna(0)
    invested = b["Total Equity"] + debt
    # tax rate from pretax / tax
    tax_rate = _safe_div(-b["Income Tax (Expense) Benefit, Net"], b["Pretax Income (Loss)"]).clip(0, 0.5).fillna(0.21)
    nopat = b["Operating Income (Loss)"] * (1 - tax_rate)
    out["ROIC"] = _safe_div(nopat, invested)
    return out


def _growth(b: pd.DataFrame) -> pd.DataFrame:
    """Compute YoY and 3Y CAGR per ticker via groupby + sort by period."""
    work = b[["Ticker", "period", "Revenue", "Net Income", "Operating Income (Loss)", "Shares (Diluted)"]].copy()
    work["fcf_base"] = b["Net Cash from Operating Activities"] + b.get("Change in Fixed Assets & Intangibles", 0)
    work["eps"] = _safe_div(work["Net Income"], work["Shares (Diluted)"])
    work["_pk"] = work["period"].map(_period_key)
    work = work.sort_values(["Ticker", "_pk"])

    out = pd.DataFrame(index=work.index)
    grp = work.groupby("Ticker", sort=False)
    for col, label in [("Revenue", "Revenue"), ("eps", "EPS"), ("fcf_base", "FCF"), ("Operating Income (Loss)", "OpInc")]:
        yoy = grp[col].pct_change(1)
        prev3 = grp[col].shift(3)
        ratio = work[col] / prev3
        # only compute CAGR when both endpoints positive (avoid complex / nan from negative roots)
        ratio = ratio.where((work[col] > 0) & (prev3 > 0))
        cagr3 = ratio.pow(1/3) - 1
        out[f"{label} Growth YoY"] = yoy
        out[f"{label} 3Y CAGR"] = cagr3

    return out.reindex(b.index)


def _health(b: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=b.index)
    debt = b["Short Term Debt"].fillna(0) + b["Long Term Debt"].fillna(0)
    cash = b["Cash, Cash Equivalents & Short Term Investments"].fillna(0)
    out["Debt/Equity"] = _safe_div(debt, b["Total Equity"])
    ebitda = b["Operating Income (Loss)"] + b.get("Depreciation & Amortization_cf", b.get("Depreciation & Amortization", 0))
    out["Net Debt/EBITDA"] = _safe_div(debt - cash, ebitda)
    out["Current Ratio"] = _safe_div(b["Total Current Assets"], b["Total Current Liabilities"])
    out["Quick Ratio"] = _safe_div(
        b["Total Current Assets"] - b["Inventories"].fillna(0),
        b["Total Current Liabilities"],
    )
    out["Interest Coverage"] = _safe_div(b["Operating Income (Loss)"], -b["Interest Expense, Net"].fillna(0))
    return out


def _cashflow(b: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=b.index)
    fcf = b["Net Cash from Operating Activities"] + b.get("Change in Fixed Assets & Intangibles", 0)
    out["FCF"] = fcf
    out["FCF Yield"] = _safe_div(fcf, b["market_cap"])
    out["FCF Margin"] = _safe_div(fcf, b["Revenue"])
    out["Cash Conversion"] = _safe_div(fcf, b["Net Income"])
    return out


def _dividend(b: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=b.index)
    div = -b["Dividends Paid"].fillna(0)  # paid out is negative in cashflow
    out["Dividend Yield"] = _safe_div(div, b["market_cap"])
    out["Payout Ratio"] = _safe_div(div, b["Net Income"])
    return out


# --- momentum (price-only) ---

def _momentum(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["ticker", "date"])

    rows = []
    for ticker, g in p.groupby("ticker", sort=False):
        g = g.set_index("date")
        if len(g) < 22:
            continue
        close = g["close"]
        latest = close.iloc[-1]
        ret = lambda n: (latest / close.iloc[-1 - n] - 1) if len(close) > n else np.nan
        ytd_start = close.loc[close.index.year == close.index[-1].year].iloc[0] if (close.index.year == close.index[-1].year).any() else np.nan
        last_252 = close.iloc[-252:]
        hi52 = last_252.max()
        lo52 = last_252.min()
        log_ret = np.log(close / close.shift(1))
        vol = log_ret.iloc[-252:].std() * np.sqrt(252)
        rows.append({
            "ticker": ticker,
            "Return 1M": ret(21),
            "Return 3M": ret(63),
            "Return 6M": ret(126),
            "Return 1Y": ret(252),
            "Return YTD": (latest / ytd_start - 1) if ytd_start and not np.isnan(ytd_start) else np.nan,
            "Dist from 52w High": (latest / hi52 - 1) if hi52 else np.nan,
            "Dist from 52w Low": (latest / lo52 - 1) if lo52 else np.nan,
            "Volatility (annualized)": vol,
        })
    return pd.DataFrame(rows)


# --- composite scores ---

def _scores(b: pd.DataFrame) -> pd.DataFrame:
    """Piotroski F-Score + Altman Z-Score. Needs prior-period data per ticker."""
    work = b.copy()
    work["_pk"] = work["period"].map(_period_key)
    work = work.sort_values(["Ticker", "_pk"])

    fcf = work["Net Cash from Operating Activities"] + work.get("Change in Fixed Assets & Intangibles", 0)

    grp = work.groupby("Ticker", group_keys=False)
    prev = grp.shift(1)

    debt = work["Short Term Debt"].fillna(0) + work["Long Term Debt"].fillna(0)
    prev_debt = prev["Short Term Debt"].fillna(0) + prev["Long Term Debt"].fillna(0)

    roa = _safe_div(work["Net Income"], work["Total Assets"])
    prev_roa = _safe_div(prev["Net Income"], prev["Total Assets"])

    cur_ratio = _safe_div(work["Total Current Assets"], work["Total Current Liabilities"])
    prev_cur_ratio = _safe_div(prev["Total Current Assets"], prev["Total Current Liabilities"])

    gm = _safe_div(work["Gross Profit"], work["Revenue"])
    prev_gm = _safe_div(prev["Gross Profit"], prev["Revenue"])

    asset_turn = _safe_div(work["Revenue"], work["Total Assets"])
    prev_asset_turn = _safe_div(prev["Revenue"], prev["Total Assets"])

    leverage = _safe_div(debt, work["Total Assets"])
    prev_leverage = _safe_div(prev_debt, prev["Total Assets"])

    ocf = work["Net Cash from Operating Activities"]

    f_score = (
        (roa > 0).astype(int)
        + (ocf > 0).astype(int)
        + (roa > prev_roa).astype(int)
        + (ocf > work["Net Income"]).astype(int)
        + (leverage < prev_leverage).astype(int)
        + (cur_ratio > prev_cur_ratio).astype(int)
        + (work["Shares (Diluted)"] <= prev["Shares (Diluted)"]).astype(int)
        + (gm > prev_gm).astype(int)
        + (asset_turn > prev_asset_turn).astype(int)
    )

    # Altman Z (manufacturing)
    wc = work["Total Current Assets"] - work["Total Current Liabilities"]
    ta = work["Total Assets"]
    z = (
        1.2 * _safe_div(wc, ta)
        + 1.4 * _safe_div(work["Retained Earnings"], ta)
        + 3.3 * _safe_div(work["Operating Income (Loss)"], ta)
        + 0.6 * _safe_div(work["market_cap"], work["Total Liabilities"])
        + 1.0 * _safe_div(work["Revenue"], ta)
    )

    out = pd.DataFrame({"Piotroski F": f_score, "Altman Z": z}, index=work.index).reindex(b.index)
    return out


# --- top-level ---

def compute_metrics(
    fundamentals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    period: Literal["annual", "ttm"] = "annual",
    latest: bool = True,
) -> pd.DataFrame:
    """Compute all metrics. See module docstring."""
    base = _build_base(fundamentals, period)
    base = _attach_price_asof(base, prices)

    # Growth + scores need full panel — compute first
    growth = _growth(base)
    scores = _scores(base)

    if latest:
        # Reduce to latest row per ticker, then overlay today's price for valuation
        idx = base.sort_values("Publish Date").groupby("Ticker").tail(1).index
        base = base.loc[idx].reset_index(drop=True)
        growth = growth.loc[idx].reset_index(drop=True)
        scores = scores.loc[idx].reset_index(drop=True)
        base = _overlay_latest_price(base, prices)

    parts = [
        base[["Ticker", "period", "Publish Date", "price", "market_cap"]].reset_index(drop=True),
        _valuation(base).reset_index(drop=True),
        _profitability(base).reset_index(drop=True),
        growth.reset_index(drop=True),
        _health(base).reset_index(drop=True),
        _cashflow(base).reset_index(drop=True),
        _dividend(base).reset_index(drop=True),
        scores.reset_index(drop=True),
    ]
    out = pd.concat(parts, axis=1)
    out = out.rename(columns={"Ticker": "ticker"})

    mom = _momentum(prices)
    out = out.merge(mom, on="ticker", how="left")

    if latest:
        out = out.drop(columns=["Publish Date"])
    return out
