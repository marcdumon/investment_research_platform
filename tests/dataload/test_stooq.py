"""Stooq provider: manual-zip acquire + normalize -> canonical prices."""
import datetime
from contextlib import contextmanager

import duckdb
import pandas as pd
import pytest

from dataload.context import IngestContext
from dataload.load import load_dataset
from dataload.providers.base import Capability
from dataload.providers.stooq import StooqProvider, _build_price_dataset, _normalize_prices

STOOQ_COLS = ['<TICKER>', '<PER>', '<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>', '<OPENINT>']
CANONICAL = {'Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Src', 'SrcId'}


def _bulk_parquet(rows: list[list], path) -> object:
    pd.DataFrame(rows, columns=STOOQ_COLS).to_parquet(path, index=False)
    return path


def _ctx(tmp_path) -> IngestContext:
    @contextmanager
    def connect():
        con = duckdb.connect(str(tmp_path / 'db.duckdb'))
        try:
            yield con
        finally:
            con.close()

    cfg = {'stooq': {'raw_dir': 'stooq/raw', 'processed_dir': 'stooq/processed',
                     'bulk_files': ['d_us_txt.zip'], 'update_file': 'data_d.txt'}}
    return IngestContext(tmp_path, cfg, connect)


def test_capabilities_prices_incremental() -> None:
    assert StooqProvider().capabilities() == {'prices': Capability(incremental=True)}


def test_normalize_produces_canonical_columns_and_values(tmp_path) -> None:
    src = _bulk_parquet([['AAPL.US', 'D', 20230101, 0, 150.0, 155.0, 148.0, 152.0, 1_000_000, 0]],
                        tmp_path / 'b.parquet')
    out = tmp_path / 'out.parquet'
    con = duckdb.connect()
    _normalize_prices(con, src, out)
    df = pd.read_parquet(out)
    assert set(df.columns) == CANONICAL
    row = df.iloc[0]
    assert row['Ticker'] == 'AAPL'           # upper + split on '.'
    assert row['SrcId'] == 'AAPL.US'         # raw stooq ticker preserved
    assert row['Src'] == 'stooq'
    assert row['Close'] == 152.0
    assert pd.Timestamp(row['Date']).date() == datetime.date(2023, 1, 1)


def test_normalize_output_loads_into_unified_prices(tmp_path) -> None:
    src = _bulk_parquet([['AAPL.US', 'D', 20230101, 0, 150.0, 155.0, 148.0, 152.0, 1_000_000, 0]],
                        tmp_path / 'b.parquet')
    out = tmp_path / 'out.parquet'
    nc = duckdb.connect()
    _normalize_prices(nc, src, out)
    nc.close()
    con = duckdb.connect(str(tmp_path / 'db.duckdb'))
    assert load_dataset(con, 'prices', out) == 1
    assert con.execute('SELECT Src FROM prices').fetchone()[0] == 'stooq'


def test_build_price_dataset_resolves_markets_through_shards_and_subcats(tmp_path) -> None:
    """Numeric shard dirs and named subcategory dirs both resolve to the right market.

    markets.csv is the seed for the universe table, so this mapping is what the
    crypto-vs-stock dedup later depends on.
    """
    raw = tmp_path / 'raw'

    def _mk(rel: str, ticker: str) -> None:
        p = raw / 'data' / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n'
                     f'{ticker},D,20230101,0,1,2,0.5,1.5,100,0\n')

    _mk('daily/us/nasdaq stocks/1/aapl.us.txt', 'AAPL')      # numeric shard
    _mk('daily/us/nasdaq stocks/2/msft.us.txt', 'MSFT')      # numeric shard
    _mk('daily/world/currencies/major/eurusd.txt', 'EURUSD')  # named subcategory

    _build_price_dataset(raw)

    markets = pd.read_csv(raw / 'markets.csv').set_index('ticker')['market']
    assert markets['aapl.us'] == 'nasdaq stocks'
    assert markets['msft.us'] == 'nasdaq stocks'
    assert markets['eurusd'] == 'currencies'
    assert (raw / 'bulk_prices.parquet').exists()


def test_produce_raises_when_bulk_zips_missing(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    (tmp_path / 'stooq' / 'raw').mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        StooqProvider().produce(ctx, ['prices'], incremental=False)
