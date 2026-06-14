"""Provider marker management.

Sources track their own progress on disk via dotfiles in `raw_dir`:
  .fetched         — bulk fetch complete
  .fetched_update  — update fetch complete
  .transformed_X   — feed X transformed
  .stored_X        — feed X stored

Each provider creates a `MarkerSet` over its raw directory and calls
`.touch(kind)` / `.is_fresh(kind, upstream)` instead of constructing paths inline.
"""
import datetime
from dataclasses import dataclass
from pathlib import Path

from dataload.state.freshness import is_fresh as _is_fresh_func


@dataclass(frozen=True)
class MarkerSet:
    """Read/write markers under one directory; freshness checks vs upstream paths."""
    root: Path

    def path(self, kind: str) -> Path:
        return self.root / f'.{kind}'

    def exists(self, kind: str) -> bool:
        return self.path(kind).exists()

    def touch(self, kind: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path(kind).touch()

    def is_fresh(self, kind: str, *upstreams: 'str | Path') -> bool:
        """True iff `kind` marker exists and is newer than every upstream.

        `upstreams` may be raw `Path`s or `str` kind names (resolved against this root).
        """
        marker = self.path(kind)
        resolved = [self.path(u) if isinstance(u, str) else u for u in upstreams]
        return _is_fresh_func(marker, *resolved)

    def age_days(self, kind: str) -> float | None:
        """Days since this marker was last touched; None if missing."""
        m = self.path(kind)
        if not m.exists():
            return None
        delta = datetime.datetime.now().timestamp() - m.stat().st_mtime
        return delta / 86400.0

    def clear(self, kind: str | None = None) -> int:
        """Delete one marker by kind, or every dotfile marker in the root.

        Returns the number of files removed.
        """
        if kind is not None:
            p = self.path(kind)
            if p.exists():
                p.unlink()
                return 1
            return 0
        n = 0
        if not self.root.exists():
            return 0
        for p in self.root.iterdir():
            if p.is_file() and p.name.startswith('.'):
                p.unlink()
                n += 1
        return n

    def clear_feed(self, feed: str) -> int:
        """Delete the bulk-or-update fetch marker plus transformed/stored markers for one feed."""
        n = 0
        fetch_name = '.fetched_update' if feed == 'update' else '.fetched'
        for name in [fetch_name, f'.transformed_{feed}', f'.stored_{feed}']:
            p = self.root / name
            if p.exists():
                p.unlink()
                n += 1
        return n
