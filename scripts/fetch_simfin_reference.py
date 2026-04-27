"""Load SimFin reference data (companies, industries) into DuckDB."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from irp.sources.simfin import SimFinFundamentalsSource
from irp.store import Store

store = Store()

for name in ["companies", "industries"]:
    print(f"Fetching {name}...", flush=True)
    dataset = SimFinFundamentalsSource(name).fetch()
    store.save(dataset, table=name)
    print(f"  {len(dataset.data):,} rows — done.", flush=True)

print("Reference data loaded.")
