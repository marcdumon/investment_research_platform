"""Data catalog: one-row-per-ticker coverage summary across all data sources.

The `catalog` table has one row per ticker (from `universe`) with columns
for every data source: Stooq price range, Yahoo price range, dividend and
split counts, SimFin fundamental period counts, company metadata flag, and
Yahoo fetch status (queried / errored).

Yahoo fetch status columns (`yahoo_prices_queried`, `yahoo_prices_error`,
`yahoo_actions_queried`, `yahoo_actions_error`) are derived from the JSON
files in `data/yahoo/raw/`. Those JSON files are the live source of truth
for the fetch pipeline; the catalog columns are a snapshot for analysis only.

`refresh()` rebuilds the table from current DB state + the JSON files.
Trigger via `uv run irp` → Steps → catalog.

`catalog()` is a read-only accessor (uses the db() singleton), consistent
with irp.query.stooq / irp.query.yahoo.
"""
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd

from irp.core.config import config
from irp.query._common import db, ticker_filter

logger = logging.getLogger(__name__)

_raw_dir = Path(config.data.root_dir) / config.providers.yahoo.raw_dir


def catalog(tickers: str | list[str] | None = None) -> pd.DataFrame:
    """Per-ticker coverage snapshot across all data sources."""
    clause, params = ticker_filter(tickers)
    where = f'WHERE {clause}' if clause else ''
    return db().execute(f'SELECT * FROM catalog {where} ORDER BY Ticker', params).df()


def _load_json_set(path: Path) -> set[str]:
    try:
        return set(json.loads(path.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()
    return bool(row and row[0])


def _empty_cte(cols: str) -> str:
    return f'SELECT {cols} WHERE FALSE'


def refresh() -> int:
    """Rebuild the `catalog` table from current DB state + Yahoo JSON files.

    Reads queried_prices.json, queried_actions.json, and error_tickers.json
    from the Yahoo raw_dir (config.providers.yahoo.raw_dir) and registers them
    as in-memory views before running the JOIN. These files are the source of
    truth for fetch status; the resulting catalog columns are a snapshot.

    Gracefully skips any of the 8 optional source tables that do not yet exist
    (useful when the DB is only partially populated).

    Opens a short-lived read-write connection (does not use the db() singleton).
    Returns the row count of the rebuilt table.
    """
    yp_queried = _load_json_set(_raw_dir / 'queried_prices.json')
    yp_errors  = _load_json_set(_raw_dir / 'error_tickers.json')
    ya_queried = _load_json_set(_raw_dir / 'queried_actions.json')

    with duckdb.connect(config.database.path) as con:
        for view, items, col in [
            ('_yp_q', yp_queried, 'yp_q'),
            ('_yp_e', yp_errors,  'yp_e'),
            ('_ya_q', ya_queried, 'ya_q'),
            ('_ya_e', yp_errors,  'ya_e'),  # actions errors share the same file
        ]:
            con.register(view, pd.DataFrame({'Ticker': sorted(items), col: True}))

        ex = {t: _table_exists(con, t) for t in [
            'prices', 'yahoo_prices', 'dividends', 'splits',
            'income', 'balance', 'cashflow', 'companies',
        ]}

        stooq_body = (
            'SELECT Ticker, MIN(Date) AS stooq_first, MAX(Date) AS stooq_last, COUNT(*) AS stooq_rows FROM prices GROUP BY Ticker'
            if ex['prices'] else
            _empty_cte('NULL::VARCHAR AS Ticker, NULL::DATE AS stooq_first, NULL::DATE AS stooq_last, 0::BIGINT AS stooq_rows')
        )
        yp_body = (
            'SELECT Ticker, MIN(Date) AS yahoo_first, MAX(Date) AS yahoo_last, COUNT(*) AS yahoo_rows FROM yahoo_prices GROUP BY Ticker'
            if ex['yahoo_prices'] else
            _empty_cte('NULL::VARCHAR AS Ticker, NULL::DATE AS yahoo_first, NULL::DATE AS yahoo_last, 0::BIGINT AS yahoo_rows')
        )
        div_body = (
            'SELECT Ticker, COUNT(*) AS div_count, MIN(Date) AS div_first, MAX(Date) AS div_last FROM dividends GROUP BY Ticker'
            if ex['dividends'] else
            _empty_cte('NULL::VARCHAR AS Ticker, 0::BIGINT AS div_count, NULL::DATE AS div_first, NULL::DATE AS div_last')
        )
        spl_body = (
            'SELECT Ticker, COUNT(*) AS split_count FROM splits GROUP BY Ticker'
            if ex['splits'] else
            _empty_cte('NULL::VARCHAR AS Ticker, 0::BIGINT AS split_count')
        )
        inc_body = (
            """SELECT Ticker,
                COUNT(*) FILTER (WHERE Period='A') AS income_A,
                COUNT(*) FILTER (WHERE Period='Q') AS income_Q,
                MAX("Fiscal Year")                  AS income_latest_yr
               FROM income GROUP BY Ticker"""
            if ex['income'] else
            _empty_cte('NULL::VARCHAR AS Ticker, 0::BIGINT AS income_A, 0::BIGINT AS income_Q, NULL::BIGINT AS income_latest_yr')
        )
        bal_body = (
            """SELECT Ticker,
                COUNT(*) FILTER (WHERE Period='A') AS balance_A,
                COUNT(*) FILTER (WHERE Period='Q') AS balance_Q
               FROM balance GROUP BY Ticker"""
            if ex['balance'] else
            _empty_cte('NULL::VARCHAR AS Ticker, 0::BIGINT AS balance_A, 0::BIGINT AS balance_Q')
        )
        cf_body = (
            """SELECT Ticker,
                COUNT(*) FILTER (WHERE Period='A') AS cashflow_A,
                COUNT(*) FILTER (WHERE Period='Q') AS cashflow_Q
               FROM cashflow GROUP BY Ticker"""
            if ex['cashflow'] else
            _empty_cte('NULL::VARCHAR AS Ticker, 0::BIGINT AS cashflow_A, 0::BIGINT AS cashflow_Q')
        )
        comp_body = (
            'SELECT DISTINCT Ticker FROM companies'
            if ex['companies'] else
            _empty_cte('NULL::VARCHAR AS Ticker')
        )

        con.execute(f"""
            CREATE OR REPLACE TABLE catalog AS
            WITH
            stooq AS ({stooq_body}),
            yp    AS ({yp_body}),
            div   AS ({div_body}),
            spl   AS ({spl_body}),
            inc   AS ({inc_body}),
            bal   AS ({bal_body}),
            cf    AS ({cf_body}),
            comp  AS ({comp_body})
            SELECT
                b.Ticker,
                b.Market,
                s.stooq_first,
                s.stooq_last,
                COALESCE(s.stooq_rows, 0)        AS stooq_rows,
                yp.yahoo_first,
                yp.yahoo_last,
                COALESCE(yp.yahoo_rows, 0)        AS yahoo_rows,
                COALESCE(q1.yp_q, FALSE)          AS yahoo_prices_queried,
                COALESCE(e1.yp_e, FALSE)          AS yahoo_prices_error,
                COALESCE(q2.ya_q, FALSE)          AS yahoo_actions_queried,
                COALESCE(e2.ya_e, FALSE)          AS yahoo_actions_error,
                COALESCE(d.div_count, 0)          AS div_count,
                d.div_first,
                d.div_last,
                COALESCE(sp.split_count, 0)       AS split_count,
                COALESCE(i.income_A, 0)           AS income_A,
                COALESCE(i.income_Q, 0)           AS income_Q,
                i.income_latest_yr,
                COALESCE(ba.balance_A, 0)         AS balance_A,
                COALESCE(ba.balance_Q, 0)         AS balance_Q,
                COALESCE(c.cashflow_A, 0)         AS cashflow_A,
                COALESCE(c.cashflow_Q, 0)         AS cashflow_Q,
                (comp.Ticker IS NOT NULL)         AS in_companies,
                CURRENT_TIMESTAMP                 AS catalog_updated_at
            FROM (SELECT DISTINCT Ticker, Market FROM universe) b
            LEFT JOIN stooq s  ON b.Ticker = s.Ticker
            LEFT JOIN yp       ON b.Ticker = yp.Ticker
            LEFT JOIN _yp_q q1 ON b.Ticker = q1.Ticker
            LEFT JOIN _yp_e e1 ON b.Ticker = e1.Ticker
            LEFT JOIN _ya_q q2 ON b.Ticker = q2.Ticker
            LEFT JOIN _ya_e e2 ON b.Ticker = e2.Ticker
            LEFT JOIN div d    ON b.Ticker = d.Ticker
            LEFT JOIN spl sp   ON b.Ticker = sp.Ticker
            LEFT JOIN inc i    ON b.Ticker = i.Ticker
            LEFT JOIN bal ba   ON b.Ticker = ba.Ticker
            LEFT JOIN cf c     ON b.Ticker = c.Ticker
            LEFT JOIN comp     ON b.Ticker = comp.Ticker
        """)
        count: int = con.execute('SELECT COUNT(*) FROM catalog').fetchone()[0]  # type: ignore[index]
    return count


def main() -> None:
    import time
    from irp.core.logging import configure_logging
    configure_logging()
    print('Refreshing data catalog...')
    t0 = time.perf_counter()
    n = refresh()
    print(f'Done — {n:,} tickers in {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
