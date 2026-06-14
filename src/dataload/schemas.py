"""Canonical table registry — single source of truth for every ingested table."""
from dataclasses import dataclass, field
from typing import Literal

Mode = Literal['merge', 'replace']


@dataclass(frozen=True, slots=True)
class TableSchema:
    """One target table's contract: columns, merge key, and load mode.

    Merge-table invariants are enforced at construction, so the schema registry
    cannot define a table whose column groups are inconsistent.
    """
    name: str
    mode: Mode
    columns: dict[str, str] = field(default_factory=dict)
    key: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode != 'merge':
            return
        groups = self.key + self.values + self.extra
        if not self.key:
            raise ValueError(f'{self.name}: a merge table needs a non-empty key')
        if len(groups) != len(set(groups)):
            raise ValueError(f'{self.name}: key/values/extra must be disjoint')
        if set(groups) != set(self.columns):
            raise ValueError(f'{self.name}: columns must equal key + values + extra')


def _merge(name: str, columns: dict[str, str], key: list[str], values: list[str],
           extra: list[str] | None = None) -> TableSchema:
    return TableSchema(name, 'merge', columns, key, values, extra or [])


def _replace(name: str) -> TableSchema:
    """Fundamentals tables are wide and source-defined; the parquet IS the schema."""
    return TableSchema(name, 'replace')


_PRICES = _merge(
    'prices',
    {'Ticker': 'VARCHAR', 'Date': 'DATE', 'Open': 'DOUBLE', 'High': 'DOUBLE', 'Low': 'DOUBLE',
     'Close': 'DOUBLE', 'Volume': 'BIGINT', 'Src': 'VARCHAR', 'SrcId': 'VARCHAR'},
    key=['Ticker', 'Date', 'Src'],
    values=['Open', 'High', 'Low', 'Close', 'Volume'],
    extra=['SrcId'],
)
_DIVIDENDS = _merge(
    'dividends',
    {'Ticker': 'VARCHAR', 'Date': 'DATE', 'Amount': 'DOUBLE', 'Src': 'VARCHAR', 'SrcId': 'VARCHAR'},
    key=['Ticker', 'Date', 'Src'],
    values=['Amount'],
    extra=['SrcId'],
)
_SPLITS = _merge(
    'splits',
    {'Ticker': 'VARCHAR', 'Date': 'DATE', 'Ratio': 'DOUBLE', 'Src': 'VARCHAR', 'SrcId': 'VARCHAR'},
    key=['Ticker', 'Date', 'Src'],
    values=['Ratio'],
    extra=['SrcId'],
)

# SimFin fundamentals + metadata: full-replace, columns defined by the source parquet.
_REPLACE_TABLES = [
    'income', 'balance', 'cashflow', 'derived',
    'income_restated', 'balance_restated', 'cashflow_restated', 'derived_restated',
    'companies',
]

SCHEMAS: dict[str, TableSchema] = {
    s.name: s for s in (_PRICES, _DIVIDENDS, _SPLITS, *(_replace(n) for n in _REPLACE_TABLES))
}
