"""Print row counts for all tables in the store."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import duckdb
import irp.config as _config

db_path = _config.load()["store"]["db_path"]

with duckdb.connect(db_path, read_only=True) as con:
    tables = [r[0] for r in con.execute("SELECT name FROM _irp_datasets ORDER BY name").fetchall()]
    for t in tables:
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        print(f"  {t:<20} {n:>14,}")
