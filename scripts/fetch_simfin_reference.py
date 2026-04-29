"""Load SimFin reference data (companies, industries) into DuckDB."""

import logging

from dotenv import load_dotenv

from irp._logging import configure
from irp.sources.simfin import SimFinFundamentalsSource, SimFinDatasetType
from irp.store import Store

load_dotenv()
configure()
logger = logging.getLogger(__name__)

_PKS: dict[SimFinDatasetType, list[str]] = {
    "companies": ["source_id"],
    "industries": ["IndustryId"],
}

store = Store()

for ref, pk in _PKS.items():
    logger.info("Fetching %s...", ref)
    dataset = SimFinFundamentalsSource(ref).fetch()
    store.upsert(dataset, table=ref, primary_key=pk)
    logger.info("  %d rows — done.", len(dataset.data))

logger.info("Reference data loaded.")
