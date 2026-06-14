"""Migrate an existing DB to the unified `prices` schema — no re-fetch needed.

Before: `prices(Ticker, SrcId, Date, Src, O, H, L, C, V)` (Stooq) and a separate
`yahoo_prices(Ticker, Date, Open..Volume)`. After: one `prices` table keyed on
`(Ticker, Date, Src)` with canonical `Open..Volume` columns.

Run once against the production DB:

    uv run python -m irp.migrations.unify_prices

Idempotent: a second run on an already-unified DB is a no-op transform.
"""
import logging

import duckdb

logger = logging.getLogger(__name__)


def _tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {r[0] for r in con.execute('SHOW TABLES').fetchall()}


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {r[0] for r in con.execute(f'DESCRIBE {table}').fetchall()}


def _date_expr(con: duckdb.DuckDBPyConnection, table: str) -> str:
    """SQL to coerce `Date` to DATE, tolerating a legacy YYYYMMDD-int column."""
    row = con.execute(
        'SELECT data_type FROM information_schema.columns '
        "WHERE table_name = ? AND column_name = 'Date'",
        [table],
    ).fetchone()
    dtype = (row[0] if row else 'DATE').upper()
    if 'INT' in dtype:
        return "strptime(CAST(Date AS VARCHAR), '%Y%m%d')::DATE"
    return 'CAST(Date AS DATE)'


def migrate(con: duckdb.DuckDBPyConnection) -> int:
    """Rebuild `prices` in the unified schema from whatever price tables exist.

    Returns the unified row count (0 if there was nothing to migrate).
    """
    tables = _tables(con)
    if 'prices' not in tables and 'yahoo_prices' not in tables:
        logger.info('no price tables found, nothing to migrate')
        return 0

    con.execute("""
        CREATE OR REPLACE TABLE _prices_unified (
            Ticker VARCHAR, Date DATE, Open DOUBLE, High DOUBLE, Low DOUBLE,
            Close DOUBLE, Volume BIGINT, Src VARCHAR, SrcId VARCHAR
        )
    """)

    if 'prices' in tables:
        cols = _columns(con, 'prices')
        dexpr = _date_expr(con, 'prices')
        ohlcv = ('O, H, L, C, CAST(V AS BIGINT)' if {'O', 'H', 'L', 'C', 'V'} <= cols
                 else 'Open, High, Low, Close, CAST(Volume AS BIGINT)')
        con.execute(f"""
            INSERT INTO _prices_unified
            SELECT Ticker, {dexpr}, {ohlcv}, COALESCE(Src, 'stooq'), COALESCE(SrcId, Ticker)
            FROM prices
        """)
        migrated = con.execute('SELECT COUNT(*) FROM prices').fetchone()[0]  # type: ignore[index]
        logger.info('migrated %s rows from prices', migrated)

    if 'yahoo_prices' in tables:
        dexpr = _date_expr(con, 'yahoo_prices')
        con.execute(f"""
            INSERT INTO _prices_unified
            SELECT Ticker, {dexpr}, Open, High, Low, Close, CAST(Volume AS BIGINT), 'yahoo', Ticker
            FROM yahoo_prices
        """)
        con.execute('DROP TABLE yahoo_prices')
        logger.info('folded yahoo_prices into unified prices')

    con.execute('DROP TABLE IF EXISTS prices')
    con.execute('ALTER TABLE _prices_unified RENAME TO prices')
    count: int = con.execute('SELECT COUNT(*) FROM prices').fetchone()[0]  # type: ignore[index]
    logger.info('unified prices: %s rows', f'{count:,}')
    return count


def _main() -> None:
    from irp.core.db import write_session
    from irp.core.logging import configure_logging
    configure_logging()
    with write_session() as con:
        n = migrate(con)
    logger.info('done — %s unified price rows', f'{n:,}')


if __name__ == '__main__':
    _main()
