
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    name: str
    data: pd.DataFrame
    schema: dict[str, str]
    source: str
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", self.data.copy())

    def validate(self) -> None:
        missing = set(self.schema) - set(self.data.columns)
        if missing:
            raise ValueError(f"Dataset '{self.name}' missing columns: {missing}")
