"""One-off: populate the form column in sec_filings from the period string.

Derives form type heuristically:
  FY / Q4 period  ->  10-K
  Q1 / Q2 / Q3   ->  10-Q

Foreign filers (20-F, 6-K) will be mislabeled; run fetch_sec_filings.py
for a full rebuild when time allows.
"""
from pathlib import Path
import logging
import sys

import pandas as pd
from dotenv import load_dotenv

from irp._logging import configure
from irp.datasets.dataset import Dataset
from irp.store import Store

load_dotenv()
configure()
logger = logging.getLogger(Path(__file__).stem)

store = Store()

if not store.exists("sec_filings"):
    logger.error("sec_filings table not found")
    sys.exit(1)

df = store.load("sec_filings").data

if "form" in df.columns and df["form"].notna().all():
    logger.info("form column already fully populated — nothing to do")
    sys.exit(0)

def _infer_form(period: str) -> str:
    p = str(period)
    if p.endswith("FY") or p.endswith("A") or p.endswith("Q4"):
        return "10-K"
    return "10-Q"

missing = df["form"].isna() if "form" in df.columns else pd.Series(True, index=df.index)
df["form"] = df.get("form", pd.Series(dtype=object))
df.loc[missing, "form"] = df.loc[missing, "period"].apply(_infer_form)

n = int(missing.sum())
logger.info(f"Patching {n} rows with inferred form types")

dataset = Dataset.from_df(df, name="sec_filings", source="sec_edgar")
store.upsert(dataset, table="sec_filings", primary_key=["ticker", "period"])
logger.info("Done.")
