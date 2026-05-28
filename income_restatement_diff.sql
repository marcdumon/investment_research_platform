
-- income: tickers/periods where row-sum differs between as-filed and restated
SELECT
  i.Ticker,
  i."Fiscal Year",
  i."Fiscal Period",
  i."Report Date",
  i."Restated Date",
  i.row_sum AS sum_filed,
  r.row_sum AS sum_restated,
  ABS(i.row_sum - r.row_sum) AS diff
FROM (
  SELECT Ticker, "Fiscal Year", "Fiscal Period", "Report Date", "Restated Date",
    COALESCE(TRY_CAST(i2."Shares (Basic)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Shares (Diluted)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Sales & Services Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Financing Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cost of Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cost of Goods & Services" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cost of Financing Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cost of Other Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Gross Profit" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Operating Income" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Operating Expenses" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Selling, General & Administrative" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Selling & Marketing" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."General & Administrative" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Research & Development" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Depreciation & Amortization" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Provision for Doubtful Accounts" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Operating Expenses" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Operating Income (Loss)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Non-Operating Income (Loss)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Interest Expense, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Interest Expense" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Interest Income" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Investment Income (Loss)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Foreign Exchange Gain (Loss)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Income (Loss) from Affiliates" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Non-Operating Income (Loss)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Pretax Income (Loss), Adj." AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Abnormal Gains (Losses)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Acquired In-Process R&D" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Merger & Acquisition Expense" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Abnormal Derivatives" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Disposal of Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Early Extinguishment of Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Asset Write-Down" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Impairment of Goodwill & Intangibles" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Sale of Business" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Legal Settlement" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Restructuring Charges" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Sale of Investments & Unrealized Investments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Insurance Settlement" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Abnormal Items" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Pretax Income (Loss)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Income Tax (Expense) Benefit, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Current Income Tax" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Deferred Income Tax" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Tax Allowance/Credit" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Income (Loss) from Affiliates, Net of Taxes" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Income (Loss) from Continuing Operations" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Extraordinary Gains (Losses)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Discontinued Operations" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Accounting Charges & Other" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Income (Loss) Incl. Minority Interest" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Minority Interest" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Income" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Preferred Dividends" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Adjustments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Income (Common)" AS DOUBLE), 0)
    AS row_sum
  FROM income i2 WHERE Period = 'A'
) i
JOIN (
  SELECT Ticker, "Fiscal Year", "Fiscal Period",
    COALESCE(TRY_CAST(r2."Shares (Basic)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Shares (Diluted)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Sales & Services Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Financing Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cost of Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cost of Goods & Services" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cost of Financing Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cost of Other Revenue" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Gross Profit" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Operating Income" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Operating Expenses" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Selling, General & Administrative" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Selling & Marketing" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."General & Administrative" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Research & Development" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Depreciation & Amortization" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Provision for Doubtful Accounts" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Operating Expenses" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Operating Income (Loss)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Non-Operating Income (Loss)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Interest Expense, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Interest Expense" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Interest Income" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Investment Income (Loss)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Foreign Exchange Gain (Loss)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Income (Loss) from Affiliates" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Non-Operating Income (Loss)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Pretax Income (Loss), Adj." AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Abnormal Gains (Losses)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Acquired In-Process R&D" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Merger & Acquisition Expense" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Abnormal Derivatives" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Disposal of Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Early Extinguishment of Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Asset Write-Down" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Impairment of Goodwill & Intangibles" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Sale of Business" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Legal Settlement" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Restructuring Charges" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Sale of Investments & Unrealized Investments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Insurance Settlement" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Abnormal Items" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Pretax Income (Loss)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Income Tax (Expense) Benefit, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Current Income Tax" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Deferred Income Tax" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Tax Allowance/Credit" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Income (Loss) from Affiliates, Net of Taxes" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Income (Loss) from Continuing Operations" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Extraordinary Gains (Losses)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Discontinued Operations" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Accounting Charges & Other" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Income (Loss) Incl. Minority Interest" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Minority Interest" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Income" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Preferred Dividends" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Adjustments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Income (Common)" AS DOUBLE), 0)
    AS row_sum
  FROM income_restated r2 WHERE Period = 'A'
) r USING (Ticker, "Fiscal Year", "Fiscal Period")
WHERE ABS(i.row_sum - r.row_sum) > 0
ORDER BY ABS(i.row_sum - r.row_sum) DESC;


-- balance: tickers/periods where row-sum differs between as-filed and restated
SELECT
  i.Ticker,
  i."Fiscal Year",
  i."Fiscal Period",
  i."Report Date",
  i."Restated Date",
  i.row_sum AS sum_filed,
  r.row_sum AS sum_restated,
  ABS(i.row_sum - r.row_sum) AS diff
FROM (
  SELECT Ticker, "Fiscal Year", "Fiscal Period", "Report Date", "Restated Date",
    COALESCE(TRY_CAST(i2."Shares (Basic)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Shares (Diluted)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cash, Cash Equivalents & Short Term Investments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cash & Cash Equivalents" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Short Term Investments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Accounts & Notes Receivable" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Accounts Receivable, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Notes Receivable, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Unbilled Revenues" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Inventories" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Raw Materials" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Work In Process" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Finished Goods" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Inventory" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Short Term Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Prepaid Expenses" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Derivative & Hedging Assets (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Assets Held-for-Sale" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Deferred Tax Assets (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Income Taxes Receivable" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Discontinued Operations (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Misc. Short Term Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Total Current Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Property, Plant & Equipment, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Property, Plant & Equipment" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Accumulated Depreciation" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Long Term Investments & Receivables" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Long Term Investments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Long Term Marketable Securities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Long Term Receivables" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Long Term Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Intangible Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Goodwill" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Intangible Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Prepaid Expense" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Deferred Tax Assets (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Derivative & Hedging Assets (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Prepaid Pension Costs" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Discontinued Operations (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Investments in Affiliates" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Misc. Long Term Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Total Noncurrent Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Total Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Payables & Accruals" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Accounts Payable" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Accrued Taxes" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Interest & Dividends Payable" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Payables & Accruals" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Short Term Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Short Term Borrowings" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Short Term Capital Leases" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Current Portion of Long Term Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Short Term Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Deferred Revenue (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Liabilities from Derivatives & Hedging (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Deferred Tax Liabilities (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Liabilities from Discontinued Operations (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Misc. Short Term Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Total Current Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Long Term Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Long Term Borrowings" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Long Term Capital Leases" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Long Term Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Accrued Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Pension Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Pensions" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Post-Retirement Benefits" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Deferred Compensation" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Deferred Revenue (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Deferred Tax Liabilities (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Liabilities from Derivatives & Hedging (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Liabilities from Discontinued Operations (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Misc. Long Term Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Total Noncurrent Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Total Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Preferred Equity" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Share Capital & Additional Paid-In Capital" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Common Stock" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Additional Paid in Capital" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Share Capital" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Treasury Stock" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Retained Earnings" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Equity" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Equity Before Minority Interest" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Minority Interest" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Total Equity" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Total Liabilities & Equity" AS DOUBLE), 0)
    AS row_sum
  FROM balance i2 WHERE Period = 'A'
) i
JOIN (
  SELECT Ticker, "Fiscal Year", "Fiscal Period",
    COALESCE(TRY_CAST(r2."Shares (Basic)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Shares (Diluted)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cash, Cash Equivalents & Short Term Investments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cash & Cash Equivalents" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Short Term Investments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Accounts & Notes Receivable" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Accounts Receivable, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Notes Receivable, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Unbilled Revenues" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Inventories" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Raw Materials" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Work In Process" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Finished Goods" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Inventory" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Short Term Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Prepaid Expenses" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Derivative & Hedging Assets (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Assets Held-for-Sale" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Deferred Tax Assets (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Income Taxes Receivable" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Discontinued Operations (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Misc. Short Term Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Total Current Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Property, Plant & Equipment, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Property, Plant & Equipment" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Accumulated Depreciation" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Long Term Investments & Receivables" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Long Term Investments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Long Term Marketable Securities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Long Term Receivables" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Long Term Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Intangible Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Goodwill" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Intangible Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Prepaid Expense" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Deferred Tax Assets (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Derivative & Hedging Assets (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Prepaid Pension Costs" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Discontinued Operations (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Investments in Affiliates" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Misc. Long Term Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Total Noncurrent Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Total Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Payables & Accruals" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Accounts Payable" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Accrued Taxes" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Interest & Dividends Payable" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Payables & Accruals" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Short Term Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Short Term Borrowings" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Short Term Capital Leases" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Current Portion of Long Term Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Short Term Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Deferred Revenue (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Liabilities from Derivatives & Hedging (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Deferred Tax Liabilities (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Liabilities from Discontinued Operations (Short Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Misc. Short Term Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Total Current Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Long Term Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Long Term Borrowings" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Long Term Capital Leases" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Long Term Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Accrued Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Pension Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Pensions" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Post-Retirement Benefits" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Deferred Compensation" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Deferred Revenue (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Deferred Tax Liabilities (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Liabilities from Derivatives & Hedging (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Liabilities from Discontinued Operations (Long Term)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Misc. Long Term Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Total Noncurrent Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Total Liabilities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Preferred Equity" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Share Capital & Additional Paid-In Capital" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Common Stock" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Additional Paid in Capital" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Share Capital" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Treasury Stock" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Retained Earnings" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Equity" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Equity Before Minority Interest" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Minority Interest" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Total Equity" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Total Liabilities & Equity" AS DOUBLE), 0)
    AS row_sum
  FROM balance_restated r2 WHERE Period = 'A'
) r USING (Ticker, "Fiscal Year", "Fiscal Period")
WHERE ABS(i.row_sum - r.row_sum) > 0
ORDER BY ABS(i.row_sum - r.row_sum) DESC;


-- cashflow: tickers/periods where row-sum differs between as-filed and restated
SELECT
  i.Ticker,
  i."Fiscal Year",
  i."Fiscal Period",
  i."Report Date",
  i."Restated Date",
  i.row_sum AS sum_filed,
  r.row_sum AS sum_restated,
  ABS(i.row_sum - r.row_sum) AS diff
FROM (
  SELECT Ticker, "Fiscal Year", "Fiscal Period", "Report Date", "Restated Date",
    COALESCE(TRY_CAST(i2."Shares (Basic)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Shares (Diluted)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Income/Starting Line" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Income" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Income from Discontinued Operations" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Adjustments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Depreciation & Amortization" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Non-Cash Items" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Stock-Based Compensation" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Deferred Income Taxes" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Non-Cash Adjustments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Change in Working Capital" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Change in Accounts Receivable" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Change in Inventories" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Change in Accounts Payable" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Change in Other" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Cash from Discontinued Operations (Operating)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Cash from Operating Activities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Change in Fixed Assets & Intangibles" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Disposition of Fixed Assets & Intangibles" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Disposition of Fixed Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Disposition of Intangible Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Acquisition of Fixed Assets & Intangibles" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Purchase of Fixed Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Acquisition of Intangible Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Change in Fixed Assets & Intangibles" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Change in Long Term Investment" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Decrease in Long Term Investment" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Increase in Long Term Investment" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Cash from Acquisitions & Divestitures" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Cash from Divestitures" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cash for Acquisition of Subsidiaries" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cash for Joint Ventures" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Cash from Other Acquisitions" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Investing Activities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Cash from Discontinued Operations (Investing)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Cash from Investing Activities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Dividends Paid" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cash from (Repayment of) Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cash from (Repayment of) Short Term Debt, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cash from (Repayment of) Long Term Debt, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Repayments of Long Term Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cash from Long Term Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Cash from (Repurchase of) Equity" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Increase in Capital Stock" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Decrease in Capital Stock" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Other Financing Activities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Cash from Discontinued Operations (Financing)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Cash from Financing Activities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Cash Before Disc. Operations and FX" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Change in Cash from Disc. Operations and Other" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Cash Before FX" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Effect of Foreign Exchange Rates" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(i2."Net Change in Cash" AS DOUBLE), 0)
    AS row_sum
  FROM cashflow i2 WHERE Period = 'A'
) i
JOIN (
  SELECT Ticker, "Fiscal Year", "Fiscal Period",
    COALESCE(TRY_CAST(r2."Shares (Basic)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Shares (Diluted)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Income/Starting Line" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Income" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Income from Discontinued Operations" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Adjustments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Depreciation & Amortization" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Non-Cash Items" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Stock-Based Compensation" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Deferred Income Taxes" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Non-Cash Adjustments" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Change in Working Capital" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Change in Accounts Receivable" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Change in Inventories" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Change in Accounts Payable" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Change in Other" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Cash from Discontinued Operations (Operating)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Cash from Operating Activities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Change in Fixed Assets & Intangibles" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Disposition of Fixed Assets & Intangibles" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Disposition of Fixed Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Disposition of Intangible Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Acquisition of Fixed Assets & Intangibles" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Purchase of Fixed Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Acquisition of Intangible Assets" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Change in Fixed Assets & Intangibles" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Change in Long Term Investment" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Decrease in Long Term Investment" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Increase in Long Term Investment" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Cash from Acquisitions & Divestitures" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Cash from Divestitures" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cash for Acquisition of Subsidiaries" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cash for Joint Ventures" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Cash from Other Acquisitions" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Investing Activities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Cash from Discontinued Operations (Investing)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Cash from Investing Activities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Dividends Paid" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cash from (Repayment of) Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cash from (Repayment of) Short Term Debt, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cash from (Repayment of) Long Term Debt, Net" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Repayments of Long Term Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cash from Long Term Debt" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Cash from (Repurchase of) Equity" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Increase in Capital Stock" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Decrease in Capital Stock" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Other Financing Activities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Cash from Discontinued Operations (Financing)" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Cash from Financing Activities" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Cash Before Disc. Operations and FX" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Change in Cash from Disc. Operations and Other" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Cash Before FX" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Effect of Foreign Exchange Rates" AS DOUBLE), 0) +
    COALESCE(TRY_CAST(r2."Net Change in Cash" AS DOUBLE), 0)
    AS row_sum
  FROM cashflow_restated r2 WHERE Period = 'A'
) r USING (Ticker, "Fiscal Year", "Fiscal Period")
WHERE ABS(i.row_sum - r.row_sum) > 0
ORDER BY ABS(i.row_sum - r.row_sum) DESC;
