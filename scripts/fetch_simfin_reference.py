"""Load SimFin reference data (companies, industries) into DuckDB."""

import logging

from dotenv import load_dotenv

from irp._logging import configure
from irp.sources.simfin import SimFinFundamentalsSource, SimFinDatasetType
from irp.store import Store

load_dotenv()
configure()
logger = logging.getLogger(__name__)

store = Store()

reference_datasets: list[SimFinDatasetType] = ["companies", "industries"]

for ref in reference_datasets:
    logger.info("Fetching %s...", ref)
    dataset = SimFinFundamentalsSource(ref).fetch()
    store.save(dataset, table=ref)
    logger.info("  %d rows — done.", len(dataset.data))

logger.info("Reference data loaded.")
