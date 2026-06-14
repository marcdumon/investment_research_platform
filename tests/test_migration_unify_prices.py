"""One-time DB migration: old prices(O,H,L,C,V) + yahoo_prices -> unified prices."""
import datetime as dt

import duckdb

from irp.migrations.unify_prices import migrate


def _old_schema_db(tmp_path, *, date_as_int: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(tmp_path / 'db.duckdb'))
    date_type = 'INTEGER' if date_as_int else 'DATE'
    date_val = '20240102' if date_as_int else "DATE '2024-01-02'"
    # old stooq prices already carried Src + SrcId
    con.execute(
        f'CREATE TABLE prices (Ticker VARCHAR, SrcId VARCHAR, Date {date_type}, Src VARCHAR, '
        'O DOUBLE, H DOUBLE, L DOUBLE, C DOUBLE, V BIGINT)'
    )
    con.execute(f"INSERT INTO prices VALUES ('T', 'T.US', {date_val}, 'stooq', 1, 2, 0.5, 1.4, 100)")
    con.execute(
        'CREATE TABLE yahoo_prices (Ticker VARCHAR, Date DATE, Open DOUBLE, High DOUBLE, '
        'Low DOUBLE, Close DOUBLE, Volume BIGINT)'
    )
    con.execute("INSERT INTO yahoo_prices VALUES ('T', DATE '2024-01-02', 1, 2, 0.5, 1.6, 100)")
    return con


def test_migrate_unifies_both_sources(tmp_path) -> None:
    con = _old_schema_db(tmp_path)
    n = migrate(con)
    assert n == 2
    by_src = dict(con.execute('SELECT Src, COUNT(*) FROM prices GROUP BY Src').fetchall())
    assert by_src == {'stooq': 1, 'yahoo': 1}
    cols = {r[0] for r in con.execute('DESCRIBE prices').fetchall()}
    assert cols == {'Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Src', 'SrcId'}
    # yahoo_prices is gone
    tables = {r[0] for r in con.execute('SHOW TABLES').fetchall()}
    assert 'yahoo_prices' not in tables


def test_migrate_preserves_values_and_srcid(tmp_path) -> None:
    con = _old_schema_db(tmp_path)
    migrate(con)
    stooq = con.execute("SELECT Close, SrcId, Date FROM prices WHERE Src = 'stooq'").fetchone()
    assert stooq[0] == 1.4
    assert stooq[1] == 'T.US'
    assert stooq[2] == dt.date(2024, 1, 2)
    yahoo = con.execute("SELECT Close, SrcId FROM prices WHERE Src = 'yahoo'").fetchone()
    assert yahoo == (1.6, 'T')  # yahoo SrcId defaults to Ticker


def test_migrate_handles_integer_dates(tmp_path) -> None:
    """Defensive: if the old prices stored Date as YYYYMMDD ints, still land DATE."""
    con = _old_schema_db(tmp_path, date_as_int=True)
    migrate(con)
    d = con.execute("SELECT Date FROM prices WHERE Src = 'stooq'").fetchone()[0]
    assert d == dt.date(2024, 1, 2)


def test_migrate_idempotent_when_already_unified(tmp_path) -> None:
    con = _old_schema_db(tmp_path)
    migrate(con)
    n = migrate(con)  # second run: already unified, no yahoo_prices
    assert n == 2
    assert dict(con.execute('SELECT Src, COUNT(*) FROM prices GROUP BY Src').fetchall()) == {'stooq': 1, 'yahoo': 1}
