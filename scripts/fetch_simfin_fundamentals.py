"""Fetch SimFin fundamentals (annual + quarterly) and merge into DuckDB tables."""

import logging

from dotenv import load_dotenv

from irp._logging import configure
from irp.sources.simfin import SimFinFundamentalsSource, SimFinDatasetType
from irp.store import Store
from irp.transforms.cleaner import Cleaner

load_dotenv()

configure()
logger = logging.getLogger(__name__)

statements: list[SimFinDatasetType] = ["income", "balance", "cashflow"] 
variants: list[tuple[str, str]]  = [("annual", "A"), ("quarterly", "Q")]

store = Store()
cleaner = Cleaner()

for variant, code in variants:
    for stmt in statements:
        logger.info("Fetching %s %s...", variant, stmt)
        dataset = SimFinFundamentalsSource(stmt, variant=variant).fetch()
        dataset = cleaner.transform(dataset)
        logger.info("  %d rows — merging into '%s'...", len(dataset.data), stmt)
        store.merge(dataset, table=stmt, delete_col="variant", delete_values=[code])
        logger.info("  Done.")

logger.info("All fundamentals loaded.")
