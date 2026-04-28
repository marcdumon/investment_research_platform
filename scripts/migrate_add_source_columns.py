"""One-time migration: add source_id and source columns to prices and fundamentals tables."""

import logging

import duckdb
from dotenv import load_dotenv

import irp.config as _config
from irp._logging import configure

load_dotenv()
configure()
logger = logging.getLogger(__name__)

db_path = _config.load()["store"]["db_path"]


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()]


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return table in {r[0] for r in con.execute("SHOW TABLES").fetchall()}


with duckdb.connect(db_path) as con:
    # --- prices ---
    if _table_exists(con, "prices"):
        cols = _columns(con, "prices")
        if "source_id" not in cols:
            con.execute("ALTER TABLE prices ADD COLUMN source_id VARCHAR")
            con.execute("""
                UPDATE prices SET source_id =
                    CASE WHEN ticker LIKE '^%' THEN lower(ticker)
                         ELSE lower(ticker) || '.us'
                    END
            """)
            logger.info("prices: added source_id")
        else:
            logger.info("prices: source_id already exists, skipped")

        if "source" not in cols:
            con.execute("ALTER TABLE prices ADD COLUMN source VARCHAR DEFAULT 'stooq'")
            logger.info("prices: added source")
        else:
            logger.info("prices: source already exists, skipped")
    else:
        logger.info("prices: table not found, skipped")

    # --- fundamentals ---
    for tbl in ("income", "balance", "cashflow"):
        if not _table_exists(con, tbl):
            logger.info("%s: table not found, skipped", tbl)
            continue

        cols = _columns(con, tbl)

        if "SimFinId" in cols:
            con.execute(f'ALTER TABLE "{tbl}" RENAME COLUMN "SimFinId" TO source_id')
            logger.info("%s: renamed SimFinId -> source_id", tbl)
        elif "source_id" in cols:
            logger.info("%s: source_id already exists, skipped rename", tbl)
        else:
            logger.info("%s: SimFinId not found, skipped rename", tbl)

        cols = _columns(con, tbl)
        if "source" not in cols:
            con.execute(f"ALTER TABLE \"{tbl}\" ADD COLUMN source VARCHAR DEFAULT 'simfin'")
            logger.info("%s: added source", tbl)
        else:
            logger.info("%s: source already exists, skipped", tbl)

logger.info("Migration complete.")
