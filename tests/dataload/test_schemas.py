"""Schema registry integrity — the single source of truth that prevents drift."""
from dataload.schemas import SCHEMAS, TableSchema


def test_prices_is_canonical_unified_schema() -> None:
    s = SCHEMAS['prices']
    assert isinstance(s, TableSchema)
    assert s.mode == 'merge'
    assert s.key == ['Ticker', 'Date', 'Src']
    assert s.values == ['Open', 'High', 'Low', 'Close', 'Volume']
    assert s.extra == ['SrcId']
    assert set(s.columns) == {'Ticker', 'Date', 'Src', 'Open', 'High', 'Low', 'Close', 'Volume', 'SrcId'}


def test_no_separate_yahoo_prices_table() -> None:
    """The whole point of the unification: one prices table, source as a column."""
    assert 'yahoo_prices' not in SCHEMAS


def test_modes_are_valid() -> None:
    for name, s in SCHEMAS.items():
        assert s.mode in ('merge', 'replace'), f'{name}: bad mode {s.mode!r}'


def test_merge_tables_fully_specify_their_columns() -> None:
    """For merge tables, columns must equal key + values + extra exactly (all disjoint).

    This invariant is what makes drift structurally impossible: a writer cannot
    emit a column the schema does not name, nor omit one it does.
    """
    for name, s in SCHEMAS.items():
        if s.mode != 'merge':
            continue
        groups = s.key + s.values + s.extra
        assert len(groups) == len(set(groups)), f'{name}: overlapping column groups'
        assert set(groups) == set(s.columns), f'{name}: columns must equal key+values+extra'
        assert s.key, f'{name}: merge table needs a non-empty key'


def test_dividends_and_splits_are_merge_keyed_by_ticker_date_src() -> None:
    for name in ('dividends', 'splits'):
        s = SCHEMAS[name]
        assert s.mode == 'merge'
        assert s.key == ['Ticker', 'Date', 'Src']


def test_fundamentals_and_companies_are_replace() -> None:
    for name in ('income', 'balance', 'cashflow', 'derived', 'companies'):
        assert SCHEMAS[name].mode == 'replace'
