import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import Any, ClassVar

import pandas as pd


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclasses.dataclass(frozen=True)
class Finding:
    """Schema definition for a single anomaly.

    Rules return DataFrames with these columns rather than lists of Finding
    instances — construction is vectorised inside each rule.
    """

    table: str
    ticker: str
    variant: str  # 'A' | 'Q' — kept str; comes from raw SimFin data
    report_date: str  # YYYY-MM-DD
    column: str  # affected column(s)
    value: float | None  # computed metric (rel_error, IQR distance, pct_change, …)
    rule: str
    detail: str
    severity: Severity

    @classmethod
    def list_columns(cls) -> list[str]:
        return [f.name for f in dataclasses.fields(cls)]

    @classmethod
    def empty_df(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=cls.list_columns())


class Rule(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    severity: ClassVar[Severity] = Severity.WARNING

    def __init__(self, **config: Any) -> None:
        """Override any class-level attribute at instantiation time."""
        for k, v in config.items():
            setattr(self, k, v)

    @abstractmethod
    def check(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return a DataFrame conforming to Finding.list_columns().
        Return Finding.empty_df() when no anomalies are found.
        Implementations must be fully vectorised — no iterrows().
        """


class Registry:
    """Explicit, testable rule registry.

    Usage:
        my_registry = Registry()

        @my_registry.register
        class MyRule(Rule): ...

        @my_registry.register(up_threshold=10.0)
        class ConfiguredRule(Rule): ...
    """

    def __init__(self) -> None:
        self._entries: list[tuple[type[Rule], dict[str, Any]]] = []

    def register(
        self,
        cls: type[Rule] | None = None,
        **config: Any,
    ) -> type[Rule] | Callable[[type[Rule]], type[Rule]]:
        def _add(c: type[Rule]) -> type[Rule]:
            if not any(existing is c for existing, _ in self._entries):
                self._entries.append((c, config))
            return c

        return _add(cls) if cls is not None else _add

    def rules(self) -> list[Rule]:
        return [cls(**cfg) for cls, cfg in self._entries]

    def clear(self) -> None:
        self._entries.clear()


default_registry = Registry()
register = default_registry.register
