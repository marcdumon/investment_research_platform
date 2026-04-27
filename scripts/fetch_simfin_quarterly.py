"""Fetch SimFin quarterly statements and merge into existing DuckDB tables."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

import duckdb
import pandas as pd
import simfin

import irp.config as _config
from irp.sources.simfin import _pandas_compat

cfg = _config.load()
DB = str((Path(__file__).parents[1] / cfg["store"]["db_path"]).resolve())

simfin.set_api_key(os.environ["SIMFIN_API_KEY"])
simfin.set_data_dir(cfg["simfin"]["data_dir"])

STATEMENTS = {
    "income": simfin.load_income,
    "balance": simfin.load_balance,
    "cashflow": simfin.load_cashflow,
}

with duckdb.connect(DB) as con:
    for table, loader in STATEMENTS.items():
        print(f"Fetching quarterly {table}...", flush=True)
        with _pandas_compat():
            df = loader(variant="quarterly", market="us")
        df = df.reset_index()

        print(f"  {len(df):,} rows — merging into '{table}'...", flush=True)
        con.register("_qdf", df)
        con.execute(f"""
            DELETE FROM "{table}"
            WHERE "Fiscal Period" IN ('Q1','Q2','Q3','Q4')
        """)
        con.execute(f'INSERT INTO "{table}" SELECT * FROM _qdf')
        total = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"  Done. Table now has {total:,} rows.", flush=True)

print("All quarterly data loaded.")
