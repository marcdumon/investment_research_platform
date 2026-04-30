"""Fetch SimFin fundamentals (annual + quarterly) and upsert into DuckDB tables."""
from pathlib import Path

import logging

from dotenv import load_dotenv

from irp._logging import configure
from irp.sources.simfin import SimFinFundamentalsSource, SimFinDatasetType
from irp.store import Store
from irp.transforms.cleaner import Cleaner

load_dotenv()
configure()
logger = logging.getLogger(Path(__file__).stem)

_PK = ["variant", "source_id", "Report Date"]
statements: list[SimFinDatasetType] = ["income", "balance", "cashflow"]
variants: list[str] = ["annual", "quarterly"]

store = Store()
cleaner = Cleaner()

for variant in variants:
    for stmt in statements:
        logger.info("Fetching %s %s...", variant, stmt)
        dataset = cleaner.transform(SimFinFundamentalsSource(stmt, variant=variant).fetch())
        logger.info("  %d rows — upserting into '%s'...", len(dataset.data), stmt)
        store.upsert(dataset, table=stmt, primary_key=_PK)
        logger.info("  Done.")

logger.info("All fundamentals loaded.")
