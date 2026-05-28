"""Backward-compat shim — delegates to simfin_annotations."""
from irp.checks.simfin_annotations import (  # noqa: F401
    load_corrections,
    load_correction_notes,
    save_corrections,
    get_all_corrections,
)
