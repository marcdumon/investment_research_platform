"""Persistent sorted-set of strings backed by a JSON file.

Used for resume-tracking across long-running fetches (e.g. tickers already
queried, tickers known to error). Re-creating the file is cheap; missing
file means empty set.
"""
import json
from pathlib import Path


class JsonSet:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> set[str]:
        if self.path.exists():
            return set(json.loads(self.path.read_text()))
        return set()

    def save(self, items: set[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(items), indent=2))
