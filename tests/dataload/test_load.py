"""Generic loader: schema-validated merge / replace into DuckDB."""
import datetime

import duckdb
import pandas as pd
import pytest

from dataload.load import load_dataset

PRICES_COLS = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Src', 'SrcId']


def _prices_df(rows: list[list]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=PRICES_COLS)


def _parquet(df: pd.DataFrame, path) -> object:
    df.to_parquet(path, index=False)
    return path


def _con(tmp_path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(tmp_path / 'db.duckdb'))


_D = datetime.date(2024, 1, 2)


def test_merge_inserts_rows(tmp_path) -> None:
    con = _con(tmp_path)
    df = _prices_df([
        ['AAPL', _D, 1.0, 2.0, 0.5, 1.5, 100, 'yahoo', 'AAPL'],
        ['MSFT', _D, 3.0, 4.0, 2.5, 3.5, 200, 'yahoo', 'MSFT'],
    ])
    n = load_dataset(con, 'prices', _parquet(df, tmp_path / 'p.parquet'))
    assert n == 2
    assert con.execute("SELECT Close FROM prices WHERE Ticker = 'AAPL'").fetchone()[0] == 1.5


def test_merge_is_idempotent_and_upserts(tmp_path) -> None:
    con = _con(tmp_path)
    p = tmp_path / 'p.parquet'
    _parquet(_prices_df([['AAPL', _D, 1, 2, 0.5, 1.5, 100, 'yahoo', 'AAPL']]), p)
    load_dataset(con, 'prices', p)
    _parquet(_prices_df([['AAPL', _D, 1, 2, 0.5, 9.9, 100, 'yahoo', 'AAPL']]), p)
    n = load_dataset(con, 'prices', p)
    assert n == 1  # upsert, not duplicate
    assert con.execute("SELECT Close FROM prices WHERE Ticker = 'AAPL'").fetchone()[0] == 9.9


def test_same_ticker_date_different_src_coexist(tmp_path) -> None:
    """The unification guarantee: stooq + yahoo rows for one Ticker/Date both persist."""
    con = _con(tmp_path)
    df = _prices_df([
        ['AAPL', _D, 1, 2, 0.5, 1.5, 100, 'yahoo', 'AAPL'],
        ['AAPL', _D, 1, 2, 0.5, 1.4, 100, 'stooq', 'AAPL.US'],
    ])
    n = load_dataset(con, 'prices', _parquet(df, tmp_path / 'p.parquet'))
    assert n == 2


def test_canonical_types_enforced_regardless_of_parquet(tmp_path) -> None:
    """Date stored as a string in the parquet still lands in a DATE column."""
    con = _con(tmp_path)
    df = _prices_df([['AAPL', '2024-01-02', 1, 2, 0.5, 1.5, 100, 'yahoo', 'AAPL']])
    load_dataset(con, 'prices', _parquet(df, tmp_path / 'p.parquet'))
    dtype = con.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name = 'prices' AND column_name = 'Date'"
    ).fetchone()[0]
    assert dtype == 'DATE'


def test_validate_rejects_missing_column(tmp_path) -> None:
    con = _con(tmp_path)
    df = _prices_df([['AAPL', _D, 1, 2, 0.5, 1.5, 100, 'yahoo', 'AAPL']]).drop(columns=['Volume'])
    with pytest.raises(ValueError, match='Volume'):
        load_dataset(con, 'prices', _parquet(df, tmp_path / 'p.parquet'))


def test_validate_rejects_unexpected_column(tmp_path) -> None:
    con = _con(tmp_path)
    df = _prices_df([['AAPL', _D, 1, 2, 0.5, 1.5, 100, 'yahoo', 'AAPL']])
    df['Bogus'] = 1
    with pytest.raises(ValueError, match='Bogus'):
        load_dataset(con, 'prices', _parquet(df, tmp_path / 'p.parquet'))


def test_replace_overwrites_table(tmp_path) -> None:
    con = _con(tmp_path)
    p = tmp_path / 'c.parquet'
    pd.DataFrame({'Ticker': ['A', 'B'], 'Name': ['Acme', 'Beta']}).to_parquet(p, index=False)
    load_dataset(con, 'companies', p)
    pd.DataFrame({'Ticker': ['C'], 'Name': ['Cee']}).to_parquet(p, index=False)
    n = load_dataset(con, 'companies', p)
    assert n == 1
    assert con.execute('SELECT Ticker FROM companies').fetchall() == [('C',)]
