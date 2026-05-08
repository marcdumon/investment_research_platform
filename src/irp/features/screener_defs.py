import pandas as pd

METRIC_DEFS: list[tuple[str, str]] = [
    # size
    ("Market Cap", "$B"),
    ("Enterprise Value", "$B"),
    # valuation
    ("P/E", "ratio"),
    ("PEG", "ratio"),
    ("Earnings Yield", "pct"),
    ("P/B", "ratio"),
    ("P/S", "ratio"),
    ("P/FCF", "ratio"),
    ("EV/EBITDA", "ratio"),
    ("EV/Sales", "ratio"),
    # profitability
    ("Gross Margin", "pct"),
    ("Operating Margin", "pct"),
    ("Net Margin", "pct"),
    ("ROE", "pct"),
    ("ROA", "pct"),
    ("ROIC", "pct"),
    # growth
    ("Revenue Growth YoY", "pct"),
    ("Revenue 3Y CAGR", "pct"),
    ("EPS Growth YoY", "pct"),
    ("EPS 3Y CAGR", "pct"),
    ("FCF Growth YoY", "pct"),
    ("FCF 3Y CAGR", "pct"),
    ("OpInc Growth YoY", "pct"),
    ("OpInc 3Y CAGR", "pct"),
    # health
    ("Debt/Equity", "ratio"),
    ("Net Debt/EBITDA", "ratio"),
    ("Current Ratio", "ratio"),
    ("Quick Ratio", "ratio"),
    ("Interest Coverage", "ratio"),
    # cash flow
    ("FCF Yield", "pct"),
    ("FCF Margin", "pct"),
    ("Cash Conversion", "ratio"),
    # dividend
    ("Dividend Yield", "pct"),
    ("Payout Ratio", "pct"),
    # momentum
    ("Return 1M", "pct"),
    ("Return 3M", "pct"),
    ("Return 6M", "pct"),
    ("Return 1Y", "pct"),
    ("Return YTD", "pct"),
    ("Dist from 52w High", "pct"),
    ("Dist from 52w Low", "pct"),
    ("Volatility (annualized)", "pct"),
    # scores
    ("Piotroski F", "int"),
    ("Altman Z", "ratio"),
]

METRIC_NAMES: list[str] = [m[0] for m in METRIC_DEFS]
METRIC_FMT: dict[str, str] = dict(METRIC_DEFS)

DEFAULT_COLS: list[str] = [
    "Market Cap",
    "P/E",
    "PEG",
    "P/B",
    "ROE",
    "Net Margin",
    "Revenue 3Y CAGR",
    "Debt/Equity",
    "FCF Yield",
    "Dividend Yield",
    "Return 1Y",
    "Piotroski F",
]

TABLE_CSS: str = """<style>
table.scr { border-collapse: collapse; font-size: 12px; color: #e0e0e0; background: #141414; }
table.scr th { text-align: right; padding: 4px 8px; border-bottom: 1px solid #555;
               position: sticky; top: 0; background: #1e1e1e; color: #e0e0e0; }
table.scr td { text-align: right; padding: 3px 8px; white-space: nowrap;
               background: #ffffff; color: #000000; }
table.scr tr:nth-child(even) td { background: #e6e6e6; }
table.scr td:first-child, table.scr th:first-child { text-align: left; font-weight: bold; }
table.scr tr:hover td { background: #2a2a2a; color: #e0e0e0; }
</style>"""


def fmt_cell(v, kind: str) -> str:
    if pd.isna(v):
        return ""
    if kind == "pct":
        return f"{v:,.1%}"
    if kind == "ratio":
        return f"{v:,.2f}"
    if kind == "int":
        return f"{int(v)}"
    if kind == "$B":
        return f"${v:,.0f}B"
    return str(v)
