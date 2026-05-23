"""SimFin column name constants for the factors package.

Import these throughout the package; never use raw column strings elsewhere.
"""

# ── Meta ──────────────────────────────────────────────────────────────
TICKER        = 'Ticker'
FISCAL_YEAR   = 'Fiscal Year'
FISCAL_PERIOD = 'Fiscal Period'
PERIOD        = 'Period'
REPORT_DATE   = 'Report Date'
PUBLISH_DATE  = 'Publish Date'

# ── Income statement ──────────────────────────────────────────────────
REVENUE          = 'Revenue'
GROSS_PROFIT     = 'Gross Profit'
OPERATING_INCOME = 'Operating Income (Loss)'
NET_INCOME       = 'Net Income'
INTEREST_EXPENSE = 'Interest Expense, Net'

# ── Balance sheet ─────────────────────────────────────────────────────
TOTAL_ASSETS               = 'Total Assets'
TOTAL_EQUITY               = 'Total Equity'
TOTAL_LIABILITIES          = 'Total Liabilities'
TOTAL_CURRENT_ASSETS       = 'Total Current Assets'
TOTAL_CURRENT_LIABILITIES  = 'Total Current Liabilities'
CASH_AND_ST_INVESTMENTS    = 'Cash, Cash Equivalents & Short Term Investments'
SHORT_TERM_DEBT            = 'Short Term Debt'
LONG_TERM_DEBT             = 'Long Term Debt'
# Shares outstanding at period end; same column name as in income statement.
SHARES_DILUTED_BAL = 'Shares (Diluted)'

# ── Cash flow statement ───────────────────────────────────────────────
CFO = 'Net Cash from Operating Activities'
CFI = 'Net Cash from Investing Activities'
# D&A in cashflow is positive; in income it is negative/absent. Use cashflow.
DA  = 'Depreciation & Amortization'

# ── Yahoo prices ──────────────────────────────────────────────────────
PRICE_TICKER = 'Ticker'
PRICE_DATE   = 'Date'
PRICE_CLOSE  = 'Close'
