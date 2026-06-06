"""Backward-compat shim — delegates to simfin_annotations."""
from irp.checks.simfin_annotations import (
    _get_all_corrections as get_all_corrections,
    _load_correction_notes as load_correction_notes,
    _load_corrections as load_corrections,
    _save_corrections as save_corrections,
)

# Re-export surface (kept so F401 does not strip the aliased re-exports).
__all__ = [
    'get_all_corrections',
    'load_correction_notes',
    'load_corrections',
    'save_corrections',
]
