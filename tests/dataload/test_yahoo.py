"""Yahoo provider: resume-tracked acquire + normalize -> prices/dividends/splits."""
import datetime

import duckdb
import pandas as pd

from dataload.load import load_dataset
from dataload.providers.base import Capability
from dataload.providers.yahoo import YahooProvider, _normalize_actions, _normalize_prices, _todo


def _write(text: str, path) -> object:
    path.write_text(text)
    return path


def test_capabilities() -> None:
    """All three participate in incremental runs: prices since last date, actions
    re-fetched in full (the merge dedups, catching new dividend/split events)."""
    caps = YahooProvider().capabilities()
    assert set(caps) == {'prices', 'dividends', 'splits'}
    assert all(c == Capability(incremental=True) for c in caps.values())


def test_todo_skips_queried_and_errored() -> None:
    ticker_map = {'AAPL': 'AAPL', 'MSFT': 'MSFT', 'T': 'T', 'BAD': 'BAD'}
    todo = _todo(ticker_map, queried={'AAPL'}, errors={'BAD'})
    assert set(todo) == {'MSFT', 'T'}


def test_normalize_prices_to_canonical(tmp_path) -> None:
    src = _write(
        'Date,Ticker,Open,High,Low,Close,Volume\n'
        '2024-01-02,AAPL,1.0,2.0,0.5,1.5,100\n',
        tmp_path / 'prices.csv',
    )
    out = tmp_path / 'p.parquet'
    con = duckdb.connect()
    _normalize_prices(con, src, out)
    df = pd.read_parquet(out)
    assert set(df.columns) == {'Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Src', 'SrcId'}
    row = df.iloc[0]
    assert row['Src'] == 'yahoo'
    assert row['SrcId'] == 'AAPL'
    assert pd.Timestamp(row['Date']).date() == datetime.date(2024, 1, 2)


def test_normalize_actions_splits_into_dividends_and_splits(tmp_path) -> None:
    src = _write(
        'Ticker,Date,Type,Value\n'
        'AAPL,2024-01-02,dividend,0.24\n'
        'AAPL,2024-06-10,split,4.0\n',
        tmp_path / 'actions.csv',
    )
    div_out = tmp_path / 'div.parquet'
    spl_out = tmp_path / 'spl.parquet'
    con = duckdb.connect()
    _normalize_actions(con, src, div_out, spl_out)

    div = pd.read_parquet(div_out)
    assert set(div.columns) == {'Ticker', 'Date', 'Amount', 'Src', 'SrcId'}
    assert div.iloc[0]['Amount'] == 0.24
    assert len(div) == 1

    spl = pd.read_parquet(spl_out)
    assert set(spl.columns) == {'Ticker', 'Date', 'Ratio', 'Src', 'SrcId'}
    assert spl.iloc[0]['Ratio'] == 4.0
    assert len(spl) == 1


def test_normalized_outputs_load_into_unified_tables(tmp_path) -> None:
    p = _write('Date,Ticker,Open,High,Low,Close,Volume\n2024-01-02,AAPL,1,2,0.5,1.5,100\n',
               tmp_path / 'prices.csv')
    a = _write('Ticker,Date,Type,Value\nAAPL,2024-01-02,dividend,0.24\nAAPL,2024-06-10,split,4.0\n',
               tmp_path / 'actions.csv')
    nc = duckdb.connect()
    _normalize_prices(nc, p, tmp_path / 'p.parquet')
    _normalize_actions(nc, a, tmp_path / 'd.parquet', tmp_path / 's.parquet')
    nc.close()

    con = duckdb.connect(str(tmp_path / 'db.duckdb'))
    assert load_dataset(con, 'prices', tmp_path / 'p.parquet') == 1
    assert load_dataset(con, 'dividends', tmp_path / 'd.parquet') == 1
    assert load_dataset(con, 'splits', tmp_path / 's.parquet') == 1
    assert con.execute("SELECT Src FROM prices").fetchone()[0] == 'yahoo'
