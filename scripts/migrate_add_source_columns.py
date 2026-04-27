"""One-time migration: add source_id and source columns to prices and fundamentals tables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

import duckdb
import irp.config as _config

db_path = _config.load()["store"]["db_path"]


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [r[0] for r in con.execute(f'DESCRIBE "{table}"').fetchall()]


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return table in [r[0] for r in con.execute("SHOW TABLES").fetchall()]


with duckdb.connect(db_path) as con:
    # --- prices ---
    if _table_exists(con, "prices"):
        cols = _columns(con, "prices")
        if "source_id" not in cols:
            con.execute("ALTER TABLE prices ADD COLUMN source_id VARCHAR")
            # Best-effort: indices keep bare lowercase ticker, others get .us suffix
            con.execute("""
                UPDATE prices SET source_id =
                    CASE WHEN ticker LIKE '^%' THEN lower(ticker)
                         ELSE lower(ticker) || '.us'
                    END
            """)
            print("prices: added source_id")
        else:
            print("prices: source_id already exists, skipped")

        if "source" not in cols:
            con.execute("ALTER TABLE prices ADD COLUMN source VARCHAR DEFAULT 'stooq'")
            print("prices: added source")
        else:
            print("prices: source already exists, skipped")
    else:
        print("prices: table not found, skipped")

    # --- fundamentals ---
    for tbl in ("income", "balance", "cashflow"):
        if not _table_exists(con, tbl):
            print(f"{tbl}: table not found, skipped")
            continue

        cols = _columns(con, tbl)

        if "SimFinId" in cols:
            con.execute(f'ALTER TABLE "{tbl}" RENAME COLUMN "SimFinId" TO source_id')
            print(f"{tbl}: renamed SimFinId -> source_id")
        elif "source_id" in cols:
            print(f"{tbl}: source_id already exists, skipped rename")
        else:
            print(f"{tbl}: SimFinId not found, skipped rename")

        cols = _columns(con, tbl)  # refresh after potential rename
        if "source" not in cols:
            con.execute(f'ALTER TABLE "{tbl}" ADD COLUMN source VARCHAR DEFAULT \'simfin\'')
            print(f"{tbl}: added source")
        else:
            print(f"{tbl}: source already exists, skipped")

print("Migration complete.")
