"""Unified prices accessor: one table, Src-filtered views."""
import duckdb

import irp.query.prices as qp
import irp.query.stooq as qs
import irp.query.yahoo as qy


def _synthetic_db(tmp_path) -> str:
    dbf = tmp_path / 't.duckdb'
    con = duckdb.connect(str(dbf))
    con.execute(
        'CREATE TABLE prices (Ticker VARCHAR, Date DATE, Open DOUBLE, High DOUBLE, Low DOUBLE, '
        'Close DOUBLE, Volume BIGINT, Src VARCHAR, SrcId VARCHAR)'
    )
    con.execute(
        "INSERT INTO prices VALUES "
        "('AAPL', DATE '2024-01-02', 1, 2, 0.5, 1.5, 100, 'yahoo', 'AAPL'), "
        "('AAPL', DATE '2024-01-02', 1, 2, 0.5, 1.4, 100, 'stooq', 'AAPL.US'), "
        "('T',    DATE '2024-01-02', 1, 2, 0.5, 9.0, 100, 'yahoo', 'T')"
    )
    con.close()
    return str(dbf)


def test_prices_src_filter(tmp_path, monkeypatch) -> None:
    dbf = _synthetic_db(tmp_path)
    monkeypatch.setattr(qp, 'db', lambda: duckdb.connect(dbf, read_only=True))

    y = qp.prices(['AAPL'], src='yahoo')
    assert len(y) == 1
    assert y.iloc[0]['Close'] == 1.5

    everything = qp.prices(['AAPL'])
    assert len(everything) == 2  # both sources coexist


def test_stooq_and_yahoo_wrappers_filter_source(tmp_path, monkeypatch) -> None:
    dbf = _synthetic_db(tmp_path)
    monkeypatch.setattr(qp, 'db', lambda: duckdb.connect(dbf, read_only=True))

    s = qs.prices(['AAPL'])
    assert set(s['Src']) == {'stooq'}
    assert s.iloc[0]['Close'] == 1.4
    assert {'Open', 'High', 'Low', 'Close', 'Volume'} <= set(s.columns)  # full names, not O/H/L/C/V

    y = qy.prices(['AAPL'])
    assert set(y['Src']) == {'yahoo'}
