"""Universe seed + the market-priority dedup that fixes the crypto-vs-stock bug."""
from contextlib import contextmanager

import duckdb
import pandas as pd

from dataload.context import IngestContext
from dataload.universe import refresh, seed, seed_from_markets


def _markets_csv(rows: list[list[str]], path) -> object:
    pd.DataFrame(rows, columns=['ticker', 'market']).to_csv(path, index=False)
    return path


def test_dedup_prefers_stock_over_crypto_regardless_of_order(tmp_path) -> None:
    """THE bug: `T` exists as t.v (crypto) and t.us (AT&T). Crypto listed first
    must NOT win — the stock market does, so AT&T survives and is fetchable."""
    mc = _markets_csv([['t.v', 'cryptocurrencies'], ['t.us', 'nyse stocks']], tmp_path / 'markets.csv')
    out = tmp_path / 'universe.csv'
    seed_from_markets(mc, out)
    row = pd.read_csv(out).query("Ticker == 'T'").iloc[0]
    assert row['Market'] == 'nyse stocks'
    assert row['stooq_ticker'] == 'T.US'
    assert row['yahoo_ticker'] == 'T'


def test_dedup_unknown_market_beats_crypto(tmp_path) -> None:
    """Crypto/fx/bonds sit in the lowest tier, below even unrecognised markets."""
    mc = _markets_csv([['x.v', 'cryptocurrencies'], ['x.zz', 'some exotic market']], tmp_path / 'markets.csv')
    out = tmp_path / 'universe.csv'
    seed_from_markets(mc, out)
    assert pd.read_csv(out).query("Ticker == 'X'").iloc[0]['Market'] == 'some exotic market'


def test_one_row_per_canonical_ticker(tmp_path) -> None:
    mc = _markets_csv([['t.v', 'cryptocurrencies'], ['t.us', 'nyse stocks'],
                       ['aapl.us', 'nasdaq stocks']], tmp_path / 'markets.csv')
    out = tmp_path / 'universe.csv'
    n = seed_from_markets(mc, out)
    df = pd.read_csv(out)
    assert n == 2
    assert df['Ticker'].is_unique


def test_yahoo_ticker_currency_mapping(tmp_path) -> None:
    mc = _markets_csv([['eurusd', 'currencies']], tmp_path / 'markets.csv')
    out = tmp_path / 'universe.csv'
    seed_from_markets(mc, out)
    assert pd.read_csv(out).iloc[0]['yahoo_ticker'] == 'EURUSD=X'


def test_seed_and_refresh_write_universe_table(tmp_path) -> None:
    @contextmanager
    def connect():
        con = duckdb.connect(str(tmp_path / 'db.duckdb'))
        try:
            yield con
        finally:
            con.close()

    ctx = IngestContext(tmp_path, {'stooq': {'raw_dir': 'stooq/raw', 'processed_dir': 'stooq/processed'}}, connect)
    (ctx.raw_dir('stooq')).mkdir(parents=True)
    _markets_csv([['t.v', 'cryptocurrencies'], ['t.us', 'nyse stocks']],
                 ctx.raw_dir('stooq') / 'markets.csv')
    seed(ctx)
    n = refresh(ctx)
    assert n == 1
    with ctx.connect() as con:
        assert con.execute("SELECT Market FROM universe WHERE Ticker = 'T'").fetchone()[0] == 'nyse stocks'
