from pathlib import Path


def _is_fresh(marker: Path, *inputs: Path) -> bool:
    """True if marker exists and is newer than all inputs."""
    if not marker.exists():
        return False
    if not all(inp.exists() for inp in inputs):
        return False
    marker_mtime = marker.stat().st_mtime
    return all(inp.stat().st_mtime <= marker_mtime for inp in inputs)
