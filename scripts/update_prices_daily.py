"""Update prices table from Stooq daily data_d.txt (stooq.com/db/ custom date export)."""

import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import irp.config as _config
from irp._logging import configure
from irp.datasets.dataset import Dataset
from irp.sources.stooq import PRICE_SCHEMA, normalize_ticker
from irp.store import Store
from irp.transforms.cleaner import Cleaner

load_dotenv()
configure()
logger = logging.getLogger(__name__)

cfg = _config.load()["stooq"]
path = Path(cfg["data_dir"]) / "data_d.txt"

if not path.exists():
    logger.error("File not found: %s", path)
    logger.error("Download from https://stooq.com/db/ (select date range, save as data_d.txt)")
    raise SystemExit(1)

logger.info("Reading %s ...", path)
df = pd.read_csv(path, header=0)
df.columns = [c.strip("<>").lower() for c in df.columns]

df = df.drop(columns=["per", "time", "openint"])
df = df.rename(columns={"ticker": "source_id", "vol": "volume"})

df["source_id"] = df["source_id"].str.lower()
df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")

for col in ("open", "high", "low", "close", "volume"):
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

df.insert(0, "ticker", df["source_id"].map(normalize_ticker))
df.insert(2, "source", "stooq")
df = df.reset_index(drop=True)

logger.info("%d rows from %s unique tickers", len(df), df["source_id"].nunique())

dataset = Dataset(name="prices", data=df, schema=PRICE_SCHEMA, source="stooq_daily")
dataset = Cleaner().transform(dataset)

Store().upsert(dataset, table="prices", primary_key=["source_id", "date"])
logger.info("Done.")
