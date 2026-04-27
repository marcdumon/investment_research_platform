"""Force-refresh SimFin fundamentals from the API and merge into DuckDB tables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from irp.sources.simfin import SimFinFundamentalsSource
from irp.store import Store
from irp.transforms.cleaner import Cleaner

STATEMENTS = ["income", "balance", "cashflow"]
VARIANTS = [("annual", "A"), ("quarterly", "Q")]

store = Store()
cleaner = Cleaner()

for variant, code in VARIANTS:
    for stmt in STATEMENTS:
        print(f"Fetching {variant} {stmt} (force refresh)...", flush=True)
        dataset = SimFinFundamentalsSource(stmt, variant=variant).fetch(refresh_days=0)
        dataset = cleaner.transform(dataset)
        print(f"  {len(dataset.data):,} rows — merging into '{stmt}'...", flush=True)
        store.merge(dataset, table=stmt, delete_col="variant", delete_values=[code])
        print(f"  Done.", flush=True)

print("All fundamentals updated.")
