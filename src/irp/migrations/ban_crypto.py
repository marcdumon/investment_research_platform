"""Purge crypto (and any non-winner stooq symbol) from an existing DB — no re-fetch.

The unified `prices` table has no market column, so crypto rows like `T.V` (which collide
with the real `T.US` = AT&T) can't be spotted directly. But once the `universe` table is
re-seeded with crypto excluded, `universe.stooq_ticker` is the whitelist of legitimate stooq
symbols — anything in `prices` (Src='stooq') whose SrcId is not in it is a banned/dedup-loser
row and gets deleted.

Run once against the production DB:

    uv run python -m irp.migrations.ban_crypto

Idempotent: a second run (universe unchanged) deletes 0 rows.
"""
import logging

import duckdb

logger = logging.getLogger(__name__)


def _tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {r[0] for r in con.execute('SHOW TABLES').fetchall()}


def migrate(con: duckdb.DuckDBPyConnection) -> int:
    """Delete stooq price rows whose SrcId is not a current universe symbol.

    Assumes `universe` has already been re-seeded with crypto excluded. Returns the
    number of deleted rows (0 if nothing to purge or required tables are missing).
    """
    tables = _tables(con)
    if 'prices' not in tables or 'universe' not in tables:
        logger.info('prices/universe table missing, nothing to purge')
        return 0

    deleted: int = con.execute("""
        DELETE FROM prices
        WHERE Src = 'stooq'
          AND SrcId NOT IN (SELECT stooq_ticker FROM universe WHERE stooq_ticker IS NOT NULL)
    """).fetchone()[0]  # type: ignore[index]
    logger.info('purged %s non-whitelisted stooq price rows', f'{deleted:,}')
    return deleted


def _main() -> None:
    from dataload import universe as ul
    from irp.core.db import write_session
    from irp.core.ingest_context import build_ingest_context
    from irp.core.logging import configure_logging

    configure_logging()
    ctx = build_ingest_context()
    logger.info('re-seeding crypto-excluded universe: %s tickers', f'{ul.seed(ctx):,}')
    logger.info('refreshed universe table: %s tickers', f'{ul.refresh(ctx):,}')
    with write_session() as con:
        n = migrate(con)
    logger.info('done — purged %s rows', f'{n:,}')


if __name__ == '__main__':
    _main()
