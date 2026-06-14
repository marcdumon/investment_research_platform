"""One-time purge: delete stooq price rows whose SrcId is not a current universe symbol."""
import duckdb

from irp.migrations.ban_crypto import migrate

_PRICES_DDL = (
    'CREATE TABLE prices (Ticker VARCHAR, Date DATE, Open DOUBLE, High DOUBLE, Low DOUBLE, '
    'Close DOUBLE, Volume BIGINT, Src VARCHAR, SrcId VARCHAR)'
)


def _db(tmp_path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(tmp_path / 'db.duckdb'))
    con.execute(_PRICES_DDL)
    con.execute("""
        INSERT INTO prices VALUES
            ('T',    DATE '2023-12-29', 16.6, 16.8, 16.6, 16.78, 33285405, 'stooq', 'T.US'),
            ('T',    DATE '2023-12-29', 0.02, 0.03, 0.02, 0.026, 22209099, 'stooq', 'T.V'),
            ('AAPL', DATE '2023-01-01', 1, 1, 1, 1, 1, 'yahoo', 'AAPL')
    """)
    con.execute('CREATE TABLE universe (Ticker VARCHAR, Market VARCHAR, stooq_ticker VARCHAR, yahoo_ticker VARCHAR)')
    con.execute("INSERT INTO universe VALUES ('T', 'nyse stocks', 'T.US', 'T')")
    return con


def test_purges_crypto_keeps_equity_and_yahoo(tmp_path) -> None:
    con = _db(tmp_path)
    assert migrate(con) == 1
    assert {r[0] for r in con.execute("SELECT SrcId FROM prices WHERE Src = 'stooq'").fetchall()} == {'T.US'}
    # yahoo rows are untouched (not stooq-scoped)
    assert con.execute("SELECT COUNT(*) FROM prices WHERE Src = 'yahoo'").fetchone()[0] == 1


def test_idempotent(tmp_path) -> None:
    con = _db(tmp_path)
    migrate(con)
    assert migrate(con) == 0


def test_no_tables_is_noop(tmp_path) -> None:
    con = duckdb.connect(str(tmp_path / 'empty.duckdb'))
    assert migrate(con) == 0
