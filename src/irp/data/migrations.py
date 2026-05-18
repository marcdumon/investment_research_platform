"""One-time DB migrations. Run via `uv run irp-migrate`."""
import duckdb

from irp.core.config import config

_DATE_COLUMNS = {
    'prices':       'Ticker, Date, O, H, L, C, V, SrcId, Src',
    'dividends':    'Ticker, Date, Amount, SrcId, Src',
    'splits':       'Ticker, Date, Ratio, SrcId, Src',
    'yahoo_prices': 'Ticker, Date, Open, High, Low, Close, Volume',
}


def _current_type(con: duckdb.DuckDBPyConnection, table: str) -> str | None:
    row = con.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name = ? AND column_name = 'Date'",
        [table],
    ).fetchone()
    return row[0] if row else None


def migrate_dates(con: duckdb.DuckDBPyConnection) -> None:
    """Convert BIGINT YYYYMMDD Date columns to DATE type. Safe to re-run."""
    for table, cols in _DATE_COLUMNS.items():
        dtype = _current_type(con, table)
        if dtype is None:
            print(f'  {table}: not found, skipping')
            continue
        if dtype == 'DATE':
            print(f'  {table}: already DATE, skipping')
            continue
        print(f'  {table}: {dtype} -> DATE')
        tmp = f'_migrate_{table}'
        select_cols = cols.replace(
            'Date',
            "STRPTIME(CAST(Date AS VARCHAR), '%Y%m%d')::DATE AS Date",
            1,
        )
        con.execute(f'ALTER TABLE {table} RENAME TO {tmp}')
        con.execute(f'CREATE TABLE {table} AS SELECT {select_cols} FROM {tmp}')
        con.execute(f'DROP TABLE {tmp}')
        print(f'  {table}: done')


def main() -> None:
    print('Running migrations...')
    with duckdb.connect(config.database.path) as con:
        migrate_dates(con)
    print('Done.')


if __name__ == '__main__':
    main()
