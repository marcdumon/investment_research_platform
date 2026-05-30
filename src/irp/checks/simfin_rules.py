from dataclasses import dataclass, field
from typing import Callable, Literal

import pandas as pd

Statement = Literal['income', 'balance', 'cashflow', 'cross']


@dataclass
class Rule:
    name: str
    statement: Statement
    lhs: str
    rhs: str
    fn: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]
    columns: tuple[str, ...] = field(default_factory=tuple)


REGISTRY: list[Rule] = []

_KEY = ['Ticker', 'Fiscal Year', 'Fiscal Period', 'Period']


def _register(name: str, statement: Statement, lhs: str, rhs: str, columns: tuple[str, ...] = ()):
    def deco(fn: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]):
        REGISTRY.append(Rule(name, statement, lhs, rhs, fn, columns))
        return fn
    return deco


def _violations(df: pd.DataFrame, lhs_col: str, rhs_col: str, tol: float = 0.005) -> pd.DataFrame:
    lhs, rhs = df[lhs_col], df[rhs_col]
    diff = lhs - rhs
    denom = pd.concat([lhs.abs(), rhs.abs()], axis=1).max(axis=1).replace(0, 1)
    rel = diff.abs() / denom
    mask = rel > tol
    return df.loc[mask, _KEY].assign(
        LHS_value=lhs[mask],
        RHS_value=rhs[mask],
        diff=diff[mask],
        rel_diff=rel[mask],
    )


# ─── Balance sheet ────────────────────────────────────────────────────

@_register('assets_split', 'balance',
          'Total Assets', 'Total Current Assets + Total Noncurrent Assets',
          columns=('Total Assets', 'Total Current Assets', 'Total Noncurrent Assets'))
def _assets_split(data):
    df = data['balance'].copy()
    df['_rhs'] = df['Total Current Assets'] + df['Total Noncurrent Assets']
    return _violations(df, 'Total Assets', '_rhs')


@_register('liab_split', 'balance',
          'Total Liabilities', 'Total Current Liabilities + Total Noncurrent Liabilities',
          columns=('Total Liabilities', 'Total Current Liabilities', 'Total Noncurrent Liabilities'))
def _liab_split(data):
    df = data['balance'].copy()
    df['_rhs'] = df['Total Current Liabilities'] + df['Total Noncurrent Liabilities']
    return _violations(df, 'Total Liabilities', '_rhs')


@_register('liab_equity_split', 'balance',
          'Total Liabilities & Equity', 'Total Liabilities + Total Equity',
          columns=('Total Liabilities & Equity', 'Total Liabilities', 'Total Equity'))
def _liab_equity_split(data):
    df = data['balance'].copy()
    df['_rhs'] = df['Total Liabilities'] + df['Total Equity']
    return _violations(df, 'Total Liabilities & Equity', '_rhs')


@_register('accounting_equation', 'balance',
          'Total Assets', 'Total Liabilities & Equity',
          columns=('Total Assets', 'Total Liabilities & Equity', 'Total Liabilities', 'Total Equity'))
def _accounting_equation(data):
    df = data['balance']
    return _violations(df, 'Total Assets', 'Total Liabilities & Equity')


# ─── Income statement ─────────────────────────────────────────────────

@_register('gross_profit', 'income',
          'Gross Profit', 'Revenue + Cost of Revenue',
          columns=('Revenue', 'Cost of Revenue', 'Gross Profit'))
def _gross_profit(data):
    df = data['income'].copy()
    df['_rhs'] = df['Revenue'] + df['Cost of Revenue']
    return _violations(df, 'Gross Profit', '_rhs')


@_register('operating_income', 'income',
          'Operating Income (Loss)', 'Gross Profit + Operating Expenses',
          columns=('Gross Profit', 'Operating Expenses', 'Operating Income (Loss)'))
def _operating_income(data):
    df = data['income'].copy()
    df['_rhs'] = df['Gross Profit'] + df['Operating Expenses']
    return _violations(df, 'Operating Income (Loss)', '_rhs')


@_register('continuing_ops', 'income',
          'Income (Loss) from Continuing Operations',
          'Pretax Income (Loss) + Income Tax (Expense) Benefit, Net',
          columns=('Pretax Income (Loss)', 'Income Tax (Expense) Benefit, Net', 'Income (Loss) from Continuing Operations'))
def _continuing_ops(data):
    df = data['income'].copy()
    df['_rhs'] = df['Pretax Income (Loss)'] + df['Income Tax (Expense) Benefit, Net']
    return _violations(df, 'Income (Loss) from Continuing Operations', '_rhs')


@_register('net_income_split', 'income',
          'Net Income',
          'Income (Loss) from Continuing Operations + Net Extraordinary Gains (Losses)',
          columns=('Income (Loss) from Continuing Operations', 'Net Extraordinary Gains (Losses)', 'Net Income'))
def _net_income_split(data):
    df = data['income'].copy()
    df['_rhs'] = (
        df['Income (Loss) from Continuing Operations']
        + df['Net Extraordinary Gains (Losses)']
    )
    return _violations(df, 'Net Income', '_rhs')


# ─── Cashflow statement ───────────────────────────────────────────────

@_register('cash_chain', 'cashflow',
          'Net Change in Cash', 'CFO + CFI + CFF + FX Effect',
          columns=('Net Cash from Operating Activities', 'Net Cash from Investing Activities',
                   'Net Cash from Financing Activities', 'Effect of Foreign Exchange Rates',
                   'Net Change in Cash'))
def _cash_chain(data):
    df = data['cashflow'].copy()
    df['_rhs'] = (
        df['Net Cash from Operating Activities']
        + df['Net Cash from Investing Activities']
        + df['Net Cash from Financing Activities']
        + df['Effect of Foreign Exchange Rates'].fillna(0)
    )
    return _violations(df, 'Net Change in Cash', '_rhs')


# ─── Cross-statement ──────────────────────────────────────────────────

@_register('ni_income_vs_cashflow', 'cross',
          'Net Income (income)', 'Net Income/Starting Line (cashflow)',
          columns=('Net Income', 'Net Income/Starting Line'))
def _ni_income_vs_cashflow(data):
    inc = data['income'][_KEY + ['Net Income']]
    cf = data['cashflow'][_KEY + ['Net Income/Starting Line']]
    m = inc.merge(cf, on=_KEY, how='inner')
    return _violations(m, 'Net Income', 'Net Income/Starting Line')


@_register('da_income_vs_cashflow', 'cross',
          '|Depreciation & Amortization| income',
          '|Depreciation & Amortization| cashflow',
          columns=('Depreciation & Amortization',))
def _da_income_vs_cashflow(data):
    inc = data['income'][_KEY + ['Depreciation & Amortization']].rename(
        columns={'Depreciation & Amortization': 'DA_income'}
    )
    cf = data['cashflow'][_KEY + ['Depreciation & Amortization']].rename(
        columns={'Depreciation & Amortization': 'DA_cashflow'}
    )
    m = inc.merge(cf, on=_KEY, how='inner')
    m['DA_income'] = m['DA_income'].abs()
    m['DA_cashflow'] = m['DA_cashflow'].abs()
    return _violations(m, 'DA_income', 'DA_cashflow')


@_register('shares_basic_cross', 'cross',
          'Shares (Basic) income', 'Shares (Basic) balance',
          columns=('Shares (Basic)',))
def _shares_basic_cross(data):
    inc = data['income'][_KEY + ['Shares (Basic)']].rename(
        columns={'Shares (Basic)': 'Shares_income'}
    )
    bal = data['balance'][_KEY + ['Shares (Basic)']].rename(
        columns={'Shares (Basic)': 'Shares_balance'}
    )
    m = inc.merge(bal, on=_KEY, how='inner')
    return _violations(m, 'Shares_income', 'Shares_balance')
