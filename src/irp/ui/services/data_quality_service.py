"""Thin wrappers over irp.checks.* for the data quality UI page."""
import pandas as pd

from irp.checks import simfin_reviews as _sfr
from irp.checks import stooq_reviews as _str
from irp.checks import simfin_runner as _sf_run
from irp.checks import stooq_runner as _stq_run
from irp.checks import stooq_inspect as _stq_inspect
from irp.checks import simfin_inspect as _sf_inspect
from irp.ui.services import universe_service


def run_simfin(skip_reviewed: bool = True, variant: str = 'A') -> pd.DataFrame:
    df = _sf_run.run(variant=variant, skip_reviewed=skip_reviewed, fetch_edgar=False)
    if df.empty:
        return df
    companies = universe_service.get_companies()[['Ticker', 'Company Name', 'Market']]
    df = df.merge(companies, on='Ticker', how='left')
    df['Company Name'] = df['Company Name'].fillna('')
    df['Market'] = df['Market'].fillna('')
    df['rel_diff_pct'] = (df['rel_diff'] * 100).round(2)
    if 'Report Date' in df.columns:
        df['Report Date'] = df['Report Date'].astype(str).str[:10]
    if 'CIK' in df.columns:
        df['CIK'] = df['CIK'].where(df['CIK'].notna(), other=None)
    return df


def run_stooq(skip_reviewed: bool = True) -> pd.DataFrame:
    return _stq_run.run(skip_reviewed=skip_reviewed)


def fetch_edgar_url(cik: int | None, report_date: str | None, period: str) -> str | None:
    """Fetch one EDGAR URL on demand (one SEC HTTP request, disk-cached 7d)."""
    if not cik or not report_date:
        return None
    from irp.checks.edgar import filing_url
    return filing_url(int(cik), report_date, period)


def inspect_simfin(rule: str, ticker: str, fy: int, fp: str, period: str) -> pd.DataFrame:
    from typing import Literal
    return _sf_inspect.inspect(rule, ticker, fy, fp, period)  # type: ignore[arg-type]


def flagged_bars(ticker: str, sample_dates: list) -> pd.DataFrame:
    return _stq_inspect.flagged_bars(ticker, [int(d) for d in sample_dates])


def stooq_figure(rule: str, ticker: str, period_str: str, sample_dates: list):
    """Plotly figure from stooq_inspect.inspect, or None."""
    import plotly.graph_objects as go
    fig = _stq_inspect.inspect(rule, ticker, period_str, [int(d) for d in sample_dates])
    return fig


def yahoo_url(ticker: str, period_str: str | None = None) -> str:
    return _stq_inspect.yahoo_url(ticker, period_str)


def add_simfin_review(ticker: str, period: str, rule: str, status: str, note: str) -> None:
    _sfr.add_review(ticker, period, rule, status, note)


def add_stooq_review(ticker: str, period: str, rule: str, status: str, note: str) -> None:
    _str.add_review(ticker, period, rule, status, note)


def add_manual_flag(ticker: str, period: str, subject: str, status: str, note: str) -> None:
    _sfr.add_flag(ticker, period, subject, status, note)


def get_all_simfin_reviews() -> pd.DataFrame:
    return _sfr.load_reviews_df()


def get_all_stooq_reviews() -> pd.DataFrame:
    return _str.load_reviews_df()


def auto_suggest_status(rel_diff: float) -> str:
    return 'ok' if abs(rel_diff) < 0.01 else 'to_check'


from irp.checks import simfin_corrections as _sc  # noqa: E402


def save_edgar_corrections(
    ticker: str,
    period: str,
    stmt_kind: str,
    edgar_values: dict[str, float],
    simfin_values: dict[str, float] | None = None,
) -> None:
    _sc.save_corrections(ticker, period, stmt_kind, edgar_values, simfin_values)


def load_edgar_corrections(ticker: str, period: str, stmt_kind: str) -> dict[str, float]:
    return _sc.load_corrections(ticker, period, stmt_kind)


def get_all_edgar_corrections() -> pd.DataFrame:
    return _sc.get_all_corrections()
