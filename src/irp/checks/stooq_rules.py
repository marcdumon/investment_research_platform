"""Stooq-price anomaly rules.

Mirrors the simfin rules pattern: @register decorator + REGISTRY list of
Rule dataclasses. Each rule.fn(con) returns a DataFrame with columns
    Ticker, Year, count, sample_dates

Year is the calendar year of the offending bars; count is the number of
bad bars in that year; sample_dates lists the first ~5 offending dates
(YYYYMMDD ints) for the inspector chart.
"""
from dataclasses import dataclass
from typing import Callable

import duckdb
import pandas as pd


@dataclass
class Rule:
    name: str
    description: str
    fn: Callable[[duckdb.DuckDBPyConnection], pd.DataFrame]


REGISTRY: list[Rule] = []


def register(name: str, description: str):
    def deco(fn: Callable[[duckdb.DuckDBPyConnection], pd.DataFrame]):
        REGISTRY.append(Rule(name, description, fn))
        return fn
    return deco


@register(
    'ohlc_inconsistent',
    'OHLC bars where High < Low, Close > High, or Close < Low.',
)
def _ohlc_inconsistent(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        SELECT
            Ticker,
            CAST(SUBSTR(CAST(Date AS VARCHAR), 1, 4) AS INTEGER) AS Year,
            COUNT(*) AS count,
            LIST(Date ORDER BY Date)[1:5] AS sample_dates
        FROM prices
        WHERE H < L OR C > H OR C < L
        GROUP BY Ticker, Year
        ORDER BY count DESC
    """).df()


@register(
    'negative_price_non_bond',
    'Negative prices for instruments where negative is not expected '
    '(excludes bonds and money-market entries, where yields can be negative).',
)
def _negative_price_non_bond(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        SELECT
            p.Ticker,
            CAST(SUBSTR(CAST(p.Date AS VARCHAR), 1, 4) AS INTEGER) AS Year,
            COUNT(*) AS count,
            LIST(p.Date ORDER BY p.Date)[1:5] AS sample_dates
        FROM prices p
        JOIN universe m ON p.Ticker = m.Ticker
        WHERE (p.O < 0 OR p.H < 0 OR p.L < 0 OR p.C < 0)
          AND m.Market NOT IN ('bonds', 'money market')
        GROUP BY p.Ticker, Year
        ORDER BY count DESC
    """).df()


@register(
    'zero_volume_trading_day',
    'Zero volume on a weekday for stocks and ETFs '
    '(trading was open but no shares traded — usually a data feed issue).',
)
def _zero_volume_trading_day(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        SELECT
            p.Ticker,
            CAST(SUBSTR(CAST(p.Date AS VARCHAR), 1, 4) AS INTEGER) AS Year,
            COUNT(*) AS count,
            LIST(p.Date ORDER BY p.Date)[1:5] AS sample_dates
        FROM prices p
        JOIN universe m ON p.Ticker = m.Ticker
        WHERE p.V = 0
          AND (m.Market LIKE '%stocks%' OR m.Market LIKE '%etfs%')
          AND dayname(strptime(CAST(p.Date AS VARCHAR), '%Y%m%d'))
              NOT IN ('Saturday', 'Sunday')
        GROUP BY p.Ticker, Year
        ORDER BY count DESC
    """).df()
