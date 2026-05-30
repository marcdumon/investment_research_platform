"""Backward-compat shim — delegates to simfin_annotations."""
from irp.checks.simfin_annotations import (  # noqa: F401
    _load_corrections as load_corrections,
    _load_correction_notes as load_correction_notes,
    _save_corrections as save_corrections,
    _get_all_corrections as get_all_corrections,
)
