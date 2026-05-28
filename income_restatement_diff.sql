SELECT
  i.Ticker,
  i."Fiscal Year",
  i."Fiscal Period",
  i."Report Date",
  i."Restated Date",
  ABS(COALESCE(TRY_CAST(i."Shares (Basic)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Shares (Basic)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Shares (Diluted)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Shares (Diluted)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Sales & Services Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Sales & Services Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Financing Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Financing Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Cost of Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Cost of Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Cost of Goods & Services" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Cost of Goods & Services" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Cost of Financing Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Cost of Financing Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Cost of Other Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Cost of Other Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Gross Profit" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Gross Profit" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Operating Income" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Operating Income" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Operating Expenses" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Operating Expenses" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Selling, General & Administrative" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Selling, General & Administrative" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Selling & Marketing" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Selling & Marketing" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."General & Administrative" AS DOUBLE),0) - COALESCE(TRY_CAST(r."General & Administrative" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Research & Development" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Research & Development" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Depreciation & Amortization" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Depreciation & Amortization" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Provision for Doubtful Accounts" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Provision for Doubtful Accounts" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Operating Expenses" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Operating Expenses" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Operating Income (Loss)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Operating Income (Loss)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Non-Operating Income (Loss)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Non-Operating Income (Loss)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Interest Expense, Net" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Interest Expense, Net" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Interest Expense" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Interest Expense" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Interest Income" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Interest Income" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Investment Income (Loss)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Investment Income (Loss)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Foreign Exchange Gain (Loss)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Foreign Exchange Gain (Loss)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Income (Loss) from Affiliates" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Income (Loss) from Affiliates" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Non-Operating Income (Loss)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Non-Operating Income (Loss)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Pretax Income (Loss), Adj." AS DOUBLE),0) - COALESCE(TRY_CAST(r."Pretax Income (Loss), Adj." AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Abnormal Gains (Losses)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Abnormal Gains (Losses)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Acquired In-Process R&D" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Acquired In-Process R&D" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Merger & Acquisition Expense" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Merger & Acquisition Expense" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Abnormal Derivatives" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Abnormal Derivatives" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Disposal of Assets" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Disposal of Assets" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Early Extinguishment of Debt" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Early Extinguishment of Debt" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Asset Write-Down" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Asset Write-Down" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Impairment of Goodwill & Intangibles" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Impairment of Goodwill & Intangibles" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Sale of Business" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Sale of Business" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Legal Settlement" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Legal Settlement" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Restructuring Charges" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Restructuring Charges" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Sale of Investments & Unrealized Investments" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Sale of Investments & Unrealized Investments" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Insurance Settlement" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Insurance Settlement" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Abnormal Items" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Abnormal Items" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Pretax Income (Loss)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Pretax Income (Loss)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Income Tax (Expense) Benefit, Net" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Income Tax (Expense) Benefit, Net" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Current Income Tax" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Current Income Tax" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Deferred Income Tax" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Deferred Income Tax" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Tax Allowance/Credit" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Tax Allowance/Credit" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Income (Loss) from Affiliates, Net of Taxes" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Income (Loss) from Affiliates, Net of Taxes" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Income (Loss) from Continuing Operations" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Income (Loss) from Continuing Operations" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Net Extraordinary Gains (Losses)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Net Extraordinary Gains (Losses)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Discontinued Operations" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Discontinued Operations" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Accounting Charges & Other" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Accounting Charges & Other" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Income (Loss) Incl. Minority Interest" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Income (Loss) Incl. Minority Interest" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Minority Interest" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Minority Interest" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Net Income" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Net Income" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Preferred Dividends" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Preferred Dividends" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Adjustments" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Adjustments" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Net Income (Common)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Net Income (Common)" AS DOUBLE),0))
    AS total_diff
FROM income i
JOIN income_restated r
  USING (Ticker, "Fiscal Year", "Fiscal Period", Period)
WHERE i.Period = 'A'
  AND (
    ABS(COALESCE(TRY_CAST(i."Shares (Basic)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Shares (Basic)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Shares (Diluted)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Shares (Diluted)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Sales & Services Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Sales & Services Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Financing Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Financing Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Cost of Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Cost of Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Cost of Goods & Services" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Cost of Goods & Services" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Cost of Financing Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Cost of Financing Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Cost of Other Revenue" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Cost of Other Revenue" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Gross Profit" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Gross Profit" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Operating Income" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Operating Income" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Operating Expenses" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Operating Expenses" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Selling, General & Administrative" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Selling, General & Administrative" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Selling & Marketing" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Selling & Marketing" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."General & Administrative" AS DOUBLE),0) - COALESCE(TRY_CAST(r."General & Administrative" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Research & Development" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Research & Development" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Depreciation & Amortization" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Depreciation & Amortization" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Provision for Doubtful Accounts" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Provision for Doubtful Accounts" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Operating Expenses" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Operating Expenses" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Operating Income (Loss)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Operating Income (Loss)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Non-Operating Income (Loss)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Non-Operating Income (Loss)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Interest Expense, Net" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Interest Expense, Net" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Interest Expense" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Interest Expense" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Interest Income" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Interest Income" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Investment Income (Loss)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Investment Income (Loss)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Foreign Exchange Gain (Loss)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Foreign Exchange Gain (Loss)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Income (Loss) from Affiliates" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Income (Loss) from Affiliates" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Non-Operating Income (Loss)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Non-Operating Income (Loss)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Pretax Income (Loss), Adj." AS DOUBLE),0) - COALESCE(TRY_CAST(r."Pretax Income (Loss), Adj." AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Abnormal Gains (Losses)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Abnormal Gains (Losses)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Acquired In-Process R&D" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Acquired In-Process R&D" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Merger & Acquisition Expense" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Merger & Acquisition Expense" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Abnormal Derivatives" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Abnormal Derivatives" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Disposal of Assets" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Disposal of Assets" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Early Extinguishment of Debt" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Early Extinguishment of Debt" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Asset Write-Down" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Asset Write-Down" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Impairment of Goodwill & Intangibles" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Impairment of Goodwill & Intangibles" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Sale of Business" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Sale of Business" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Legal Settlement" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Legal Settlement" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Restructuring Charges" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Restructuring Charges" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Sale of Investments & Unrealized Investments" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Sale of Investments & Unrealized Investments" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Insurance Settlement" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Insurance Settlement" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Abnormal Items" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Abnormal Items" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Pretax Income (Loss)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Pretax Income (Loss)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Income Tax (Expense) Benefit, Net" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Income Tax (Expense) Benefit, Net" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Current Income Tax" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Current Income Tax" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Deferred Income Tax" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Deferred Income Tax" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Tax Allowance/Credit" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Tax Allowance/Credit" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Income (Loss) from Affiliates, Net of Taxes" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Income (Loss) from Affiliates, Net of Taxes" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Income (Loss) from Continuing Operations" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Income (Loss) from Continuing Operations" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Net Extraordinary Gains (Losses)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Net Extraordinary Gains (Losses)" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Discontinued Operations" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Discontinued Operations" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Accounting Charges & Other" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Accounting Charges & Other" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Income (Loss) Incl. Minority Interest" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Income (Loss) Incl. Minority Interest" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Minority Interest" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Minority Interest" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Net Income" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Net Income" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Preferred Dividends" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Preferred Dividends" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Other Adjustments" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Other Adjustments" AS DOUBLE),0)) +
    ABS(COALESCE(TRY_CAST(i."Net Income (Common)" AS DOUBLE),0) - COALESCE(TRY_CAST(r."Net Income (Common)" AS DOUBLE),0))
  ) > 0
ORDER BY total_diff DESC;
