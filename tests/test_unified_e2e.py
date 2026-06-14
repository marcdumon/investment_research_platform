"""End-to-end on a synthetic DB: dataload writes the unified schema, the
migrated irp downstream readers consume it. Proves the rewrite hangs together
without the production database."""
import datetime as dt

import duckdb
import pandas as pd

from dataload import universe as ul
from dataload.load import load_dataset

_PRICE_COLS = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Src', 'SrcId']


def _build_db(tmp_path) -> duckdb.DuckDBPyConnection:
    db = tmp_path / 'irp.duckdb'

    # universe with the T collision: t.v (crypto) listed before t.us (AT&T)
    mc = tmp_path / 'markets.csv'
    pd.DataFrame([['t.v', 'cryptocurrencies'], ['t.us', 'nyse stocks'], ['aapl.us', 'nasdaq stocks']],
                 columns=['ticker', 'market']).to_csv(mc, index=False)
    uni_csv = tmp_path / 'universe.csv'
    ul.seed_from_markets(mc, uni_csv)
    uni = pd.read_csv(uni_csv)

    con = duckdb.connect(str(db))
    con.register('u', uni)
    con.execute('CREATE TABLE universe AS SELECT * FROM u')
    con.unregister('u')

    d = dt.date(2024, 1, 2)  # a Tuesday
    prices = pd.DataFrame([
        ['T', d, 1.0, 2.0, 0.5, 1.5, 100, 'stooq', 'T.US'],
        ['T', d, 1.0, 2.0, 0.5, 1.6, 100, 'yahoo', 'T'],
        ['AAPL', d, 1.0, 4.0, 0.5, 9.9, 0, 'stooq', 'AAPL.US'],  # Close>High + zero volume
    ], columns=_PRICE_COLS)
    pq = tmp_path / 'p.parquet'
    prices.to_parquet(pq, index=False)
    load_dataset(con, 'prices', pq)
    return con


def test_t_bug_fixed_end_to_end(tmp_path) -> None:
    con = _build_db(tmp_path)
    # universe: the stock won, not the crypto
    assert con.execute("SELECT Market FROM universe WHERE Ticker = 'T'").fetchone()[0] == 'nyse stocks'
    # and AT&T has Yahoo prices in the unified table (would have been excluded as crypto before)
    assert con.execute("SELECT COUNT(*) FROM prices WHERE Ticker = 'T' AND Src = 'yahoo'").fetchone()[0] == 1


def test_both_sources_coexist_in_one_table(tmp_path) -> None:
    con = _build_db(tmp_path)
    by_src = dict(con.execute('SELECT Src, COUNT(*) FROM prices GROUP BY Src').fetchall())
    assert by_src == {'stooq': 2, 'yahoo': 1}


def test_migrated_stooq_rules_run_on_unified_schema(tmp_path) -> None:
    con = _build_db(tmp_path)
    from irp.checks.stooq_rules import REGISTRY

    results = {r.name: r.fn(con) for r in REGISTRY}

    ohlc = results['ohlc_inconsistent']
    assert 'AAPL' in set(ohlc['Ticker'])             # Close 9.9 > High 4.0
    assert int(ohlc.iloc[0]['sample_dates'][0]) == 20240102  # int YYYYMMDD preserved

    zero_vol = results['zero_volume_trading_day']
    assert 'AAPL' in set(zero_vol['Ticker'])         # Volume 0 on a weekday stock


def test_migrated_panel_and_catalog_sql(tmp_path) -> None:
    con = _build_db(tmp_path)
    # panel pulls Yahoo closes from the unified table
    panel = con.execute(
        "SELECT Ticker, Date, Close, Volume FROM prices WHERE Src = 'yahoo' ORDER BY Ticker, Date"
    ).df()
    assert set(panel['Ticker']) == {'T'}

    # catalog stooq/yahoo coverage bodies
    stooq_cov = con.execute(
        "SELECT Ticker, COUNT(*) n FROM prices WHERE Src = 'stooq' GROUP BY Ticker"
    ).df()
    assert set(stooq_cov['Ticker']) == {'T', 'AAPL'}


def test_stooq_inspect_bars_bridge(tmp_path) -> None:
    """The inspector's column/date bridge: DATE -> int YYYYMMDD, Open..Volume -> O..V."""
    con = _build_db(tmp_path)
    row = con.execute(
        "SELECT CAST(strftime(Date, '%Y%m%d') AS INTEGER) AS Date, "
        "Open AS O, High AS H, Low AS L, Close AS C, Volume AS V "
        "FROM prices WHERE Src = 'stooq' AND Ticker = 'AAPL'"
    ).df()
    assert int(row.iloc[0]['Date']) == 20240102
    assert list(row.columns) == ['Date', 'O', 'H', 'L', 'C', 'V']
