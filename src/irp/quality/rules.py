from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class Rule:
    name: str
    statement: str           # 'income' | 'balance' | 'cashflow' | 'cross'
    lhs: str
    rhs: str
    fn: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]


REGISTRY: list[Rule] = []

_KEY = ['Ticker', 'Fiscal Year', 'Fiscal Period', 'Period']


def register(name: str, statement: str, lhs: str, rhs: str):
    def deco(fn: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]):
        REGISTRY.append(Rule(name, statement, lhs, rhs, fn))
        return fn
    return deco


def violations(df: pd.DataFrame, lhs_col: str, rhs_col: str, tol: float = 0.005) -> pd.DataFrame:
    """Return rows where |lhs - rhs| / max(|lhs|, |rhs|) > tol. Skips NaN."""
    lhs, rhs = df[lhs_col], df[rhs_col]
    diff = lhs - rhs
    denom = pd.concat([lhs.abs(), rhs.abs()], axis=1).max(axis=1).replace(0, 1)
    rel = diff.abs() / denom
    mask = rel > tol  # NaN > tol is False, so NaN rows are skipped
    return df.loc[mask, _KEY].assign(
        LHS_value=lhs[mask],
        RHS_value=rhs[mask],
        diff=diff[mask],
        rel_diff=rel[mask],
    )


# ─── Balance sheet ────────────────────────────────────────────────────

@register('assets_split', 'balance',
          'Total Assets', 'Total Current Assets + Total Noncurrent Assets')
def _assets_split(data):
    df = data['balance'].copy()
    df['_rhs'] = df['Total Current Assets'] + df['Total Noncurrent Assets']
    return violations(df, 'Total Assets', '_rhs')


@register('liab_split', 'balance',
          'Total Liabilities', 'Total Current Liabilities + Total Noncurrent Liabilities')
def _liab_split(data):
    df = data['balance'].copy()
    df['_rhs'] = df['Total Current Liabilities'] + df['Total Noncurrent Liabilities']
    return violations(df, 'Total Liabilities', '_rhs')


@register('liab_equity_split', 'balance',
          'Total Liabilities & Equity', 'Total Liabilities + Total Equity')
def _liab_equity_split(data):
    df = data['balance'].copy()
    df['_rhs'] = df['Total Liabilities'] + df['Total Equity']
    return violations(df, 'Total Liabilities & Equity', '_rhs')


@register('accounting_equation', 'balance',
          'Total Assets', 'Total Liabilities & Equity')
def _accounting_equation(data):
    df = data['balance']
    return violations(df, 'Total Assets', 'Total Liabilities & Equity')


# ─── Income statement ─────────────────────────────────────────────────

@register('gross_profit', 'income',
          'Gross Profit', 'Revenue + Cost of Revenue')
def _gross_profit(data):
    # SimFin stores Cost of Revenue as a negative number, so we ADD.
    df = data['income'].copy()
    df['_rhs'] = df['Revenue'] + df['Cost of Revenue']
    return violations(df, 'Gross Profit', '_rhs')


@register('operating_income', 'income',
          'Operating Income (Loss)', 'Gross Profit + Operating Expenses')
def _operating_income(data):
    # SimFin stores Operating Expenses as a negative number, so we ADD.
    df = data['income'].copy()
    df['_rhs'] = df['Gross Profit'] + df['Operating Expenses']
    return violations(df, 'Operating Income (Loss)', '_rhs')


@register('continuing_ops', 'income',
          'Income (Loss) from Continuing Operations',
          'Pretax Income (Loss) + Income Tax (Expense) Benefit, Net')
def _continuing_ops(data):
    # SimFin convention: tax is stored signed (negative when expense), so ADD.
    df = data['income'].copy()
    df['_rhs'] = df['Pretax Income (Loss)'] + df['Income Tax (Expense) Benefit, Net']
    return violations(df, 'Income (Loss) from Continuing Operations', '_rhs')


@register('net_income_split', 'income',
          'Net Income',
          'Income (Loss) from Continuing Operations + Net Extraordinary Gains (Losses)')
def _net_income_split(data):
    df = data['income'].copy()
    df['_rhs'] = (
        df['Income (Loss) from Continuing Operations']
        + df['Net Extraordinary Gains (Losses)']
    )
    return violations(df, 'Net Income', '_rhs')


# ─── Cashflow statement ───────────────────────────────────────────────

@register('cash_chain', 'cashflow',
          'Net Change in Cash', 'CFO + CFI + CFF')
def _cash_chain(data):
    df = data['cashflow'].copy()
    df['_rhs'] = (
        df['Net Cash from Operating Activities']
        + df['Net Cash from Investing Activities']
        + df['Net Cash from Financing Activities']
    )
    return violations(df, 'Net Change in Cash', '_rhs')


# ─── Cross-statement ──────────────────────────────────────────────────

@register('ni_income_vs_cashflow', 'cross',
          'Net Income (income)', 'Net Income/Starting Line (cashflow)')
def _ni_income_vs_cashflow(data):
    inc = data['income'][_KEY + ['Net Income']]
    cf = data['cashflow'][_KEY + ['Net Income/Starting Line']]
    m = inc.merge(cf, on=_KEY, how='inner')
    return violations(m, 'Net Income', 'Net Income/Starting Line')


@register('da_income_vs_cashflow', 'cross',
          '|Depreciation & Amortization| income',
          '|Depreciation & Amortization| cashflow')
def _da_income_vs_cashflow(data):
    # Income stores D&A as expense (negative). Cashflow stores it as positive
    # (add-back in operating activities). Compare magnitudes.
    inc = data['income'][_KEY + ['Depreciation & Amortization']].rename(
        columns={'Depreciation & Amortization': 'DA_income'}
    )
    cf = data['cashflow'][_KEY + ['Depreciation & Amortization']].rename(
        columns={'Depreciation & Amortization': 'DA_cashflow'}
    )
    m = inc.merge(cf, on=_KEY, how='inner')
    m['DA_income'] = m['DA_income'].abs()
    m['DA_cashflow'] = m['DA_cashflow'].abs()
    return violations(m, 'DA_income', 'DA_cashflow')


@register('shares_basic_cross', 'cross',
          'Shares (Basic) income', 'Shares (Basic) balance')
def _shares_basic_cross(data):
    inc = data['income'][_KEY + ['Shares (Basic)']].rename(
        columns={'Shares (Basic)': 'Shares_income'}
    )
    bal = data['balance'][_KEY + ['Shares (Basic)']].rename(
        columns={'Shares (Basic)': 'Shares_balance'}
    )
    m = inc.merge(bal, on=_KEY, how='inner')
    return violations(m, 'Shares_income', 'Shares_balance')
