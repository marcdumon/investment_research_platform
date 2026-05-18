"""One-time migration: convert BIGINT YYYYMMDD Date columns to DATE type.

Tables affected: prices, dividends, splits, yahoo_prices.
Safe to re-run — skips tables where Date is already DATE.
"""
import duckdb

from irp.core.config import config

_COLUMNS = {
    'prices':      'Ticker, Date, O, H, L, C, V, SrcId, Src',
    'dividends':   'Ticker, Date, Amount, SrcId, Src',
    'splits':      'Ticker, Date, Ratio, SrcId, Src',
    'yahoo_prices': 'Ticker, Date, Open, High, Low, Close, Volume',
}


def _current_type(con: duckdb.DuckDBPyConnection, table: str) -> str | None:
    row = con.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name = ? AND column_name = 'Date'",
        [table],
    ).fetchone()
    return row[0] if row else None


def migrate(con: duckdb.DuckDBPyConnection) -> None:
    for table, cols in _COLUMNS.items():
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


if __name__ == '__main__':
    with duckdb.connect(config.database.path) as con:
        migrate(con)
    print('Migration complete.')
