from collections import defaultdict
import logging
from dataclasses import dataclass
from typing import Literal, Sequence

import duckdb
import requests
import simfin as sf

from irp.core.config import config
from irp.core.freshness import is_fresh

logger = logging.getLogger(__name__)

root_dir = config.data.root_dir
simfin_cfg = config.providers.simfin
raw_dir = root_dir / simfin_cfg.raw_dir
processed_dir = root_dir / simfin_cfg.processed_dir

FundamentalsNames = Literal['income', 'balance', 'cashflow']
FundamentalsVariant = Literal['annual', 'quarterly', 'ttm']
SharepricesVariant = Literal['daily', 'latest']
Market = Literal['us', 'de', 'ca', 'cn']


@dataclass(frozen=True)
class FundamentalsDataset:
    name: Literal['income', 'balance', 'cashflow']
    variant: FundamentalsVariant
    market: Market
    refresh_days: int = simfin_cfg.refresh_days_fundamentals


@dataclass(frozen=True)
class SharepricesDataset:
    variant: SharepricesVariant
    market: Market
    name: Literal['shareprices'] = 'shareprices'
    refresh_days: int = simfin_cfg.refresh_days_shareprices


@dataclass(frozen=True)
class CompaniesDataset:
    market: Market
    name: Literal['companies'] = 'companies'
    variant: None = None
    refresh_days: int = simfin_cfg.refresh_days_meta


@dataclass(frozen=True)
class MetaDataset:
    name: Literal['markets', 'industries']
    variant: None = None
    market: None = None
    refresh_days: int = simfin_cfg.refresh_days_meta


SimFinDataset = (
    FundamentalsDataset | SharepricesDataset | CompaniesDataset | MetaDataset
)


_FUNDAMENTALS_NAMES: list[FundamentalsNames] = ['income', 'balance', 'cashflow']
_FUNDAMENTALS_VARIANTS: list[FundamentalsVariant] = ['annual', 'quarterly']
_SHAREPRICES_VARIANTS: list[SharepricesVariant] = ['daily']
_MARKETS: list[Market] = ['us', 'de']
_META_NAMES: list[Literal['markets', 'industries']] = ['markets', 'industries']

BULK_DATASETS: Sequence[SimFinDataset] = (
    [
        FundamentalsDataset(name, variant, market)
        for name in _FUNDAMENTALS_NAMES
        for variant in _FUNDAMENTALS_VARIANTS
        for market in _MARKETS
    ]
    + [
        SharepricesDataset(variant, market)
        for variant in _SHAREPRICES_VARIANTS
        for market in _MARKETS
    ]
    + [CompaniesDataset(market) for market in _MARKETS]
    + [MetaDataset(name) for name in _META_NAMES]
)


def _transform_fundamentals(conn: duckdb.DuckDBPyConnection) -> None:
    _FUNDAMENTALS = {'income', 'balance', 'cashflow'}
    _PERIOD = {'annual': 'A', 'quarterly': 'Q', 'ttm': 'TTM'}

    statement_files: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for f in sorted(raw_dir.glob('*.csv')):
        parts = f.stem.split('-')
        if len(parts) < 3:
            continue
        market, statement, variant = parts[0], parts[1], parts[2]
        if statement not in _FUNDAMENTALS or variant not in _PERIOD:
            continue
        statement_files[statement].append((market, variant, str(f)))

    for statement, files in statement_files.items():
        union_sql = ' UNION ALL '.join(
            f"SELECT *, '{market}' AS Market, '{_PERIOD[variant]}' AS Period "
            f"FROM read_csv('{path}', delim=';')"
            for market, variant, path in files
        )
        out = processed_dir / f'fundamentals_{statement}.csv'
        conn.execute(f"""
            COPY (
                SELECT
                    Ticker,
                    SimFinId AS SrcId,
                    'simfin' AS Src,
                    * EXCLUDE (Ticker, SimFinId)
                FROM ({union_sql})
            ) TO '{out}' (FORMAT CSV, HEADER)
        """)
        logger.debug('Wrote %s', out)


def _transform_prices(conn: duckdb.DuckDBPyConnection) -> None:
    prices_files = sorted(raw_dir.glob('*-shareprices-*.csv'))
    if not prices_files:
        return

    src = ' UNION ALL '.join(
        f"SELECT * FROM read_csv('{f}', delim=';')" for f in prices_files
    )

    prices_out = processed_dir / 'prices.csv'
    conn.execute(f"""
        COPY (
            SELECT
                Ticker,
                Date,
                Open             AS O,
                High             AS H,
                Low              AS L,
                Close            AS C,
                Volume           AS V,
                "Adj. Close"     AS AdjClose,
                SimFinId         AS SrcId,
                'simfin'         AS Src
            FROM ({src})
        ) TO '{prices_out}' (FORMAT CSV, HEADER)
    """)
    logger.debug('Wrote %s', prices_out)

    divs_out = processed_dir / 'prices_dividends.csv'
    conn.execute(f"""
        COPY (
            SELECT
                Date,
                Ticker,
                Dividend,
                "Shares Outstanding",
                SimFinId         AS SrcId,
                'simfin'         AS Src
            FROM ({src})
        ) TO '{divs_out}' (FORMAT CSV, HEADER)
    """)
    logger.debug('Wrote %s', divs_out)


def _transform_companies(conn: duckdb.DuckDBPyConnection) -> None:
    company_files = sorted(raw_dir.glob('*-companies.csv'))
    industries_file = raw_dir / 'industries.csv'
    if not company_files or not industries_file.exists():
        return

    companies_union = ' UNION ALL '.join(
        f"SELECT * FROM read_csv('{f}', delim=';')" for f in company_files
    )

    out = processed_dir / 'companies.csv'
    conn.execute(f"""
        COPY (
            SELECT
                c.Ticker,
                c."Company Name",
                i.Industry,
                i.Sector,
                c."Business Summary",
                c."End of financial year (month)",
                c."Number Employees",
                c.CIK,
                c.ISIN,
                c."Main Currency",
                c.IndustryId,
                c.Market,
                c.SimFinId AS SrcId
            FROM ({companies_union}) c
            LEFT JOIN read_csv('{industries_file}', delim=';') i
                ON c.IndustryId = i.IndustryId
        ) TO '{out}' (FORMAT CSV, HEADER)
    """)
    logger.debug('Wrote %s', out)


def _store_fundamentals(con: duckdb.DuckDBPyConnection) -> None:
    for statement in ('income', 'balance', 'cashflow'):
        src = processed_dir / f'fundamentals_{statement}.csv'
        if not src.exists():
            continue
        table = f'fundamentals_{statement}'
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM read_csv_auto('{src}') LIMIT 0")
        con.execute(f"""
            MERGE INTO {table} t
            USING (
                SELECT DISTINCT ON (SrcId, "Fiscal Year", "Fiscal Period") *
                FROM read_csv_auto('{src}')
            ) s
            ON t.SrcId = s.SrcId
            AND t."Fiscal Year" = s."Fiscal Year"
            AND t."Fiscal Period" = s."Fiscal Period"
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)
        logger.debug('Stored %s', table)


def _store_prices_simfin(con: duckdb.DuckDBPyConnection) -> None:
    src = processed_dir / 'prices.csv'
    if not src.exists():
        return
    key_cols = ['Ticker', 'SrcId', 'Date', 'Src']
    update_cols = ['O', 'H', 'L', 'C', 'V', 'AdjClose']
    insert_cols = ['Ticker', 'Date', 'O', 'H', 'L', 'C', 'V', 'AdjClose', 'SrcId', 'Src']
    on_clause = ' AND '.join(f't.{c} = s.{c}' for c in key_cols)
    update_set = ', '.join(f'{c} = s.{c}' for c in update_cols)
    insert_cols_sql = ', '.join(insert_cols)
    insert_vals_sql = ', '.join(f's.{c}' for c in insert_cols)
    # Cast DATE -> INTEGER (YYYYMMDD) to match stooq's prices table schema
    src_sql = f"""
        SELECT Ticker, CAST(strftime(Date, '%Y%m%d') AS INTEGER) AS Date,
               O, H, L, C, V, AdjClose, SrcId, Src
        FROM read_csv_auto('{src}')
    """
    con.execute(f"CREATE TABLE IF NOT EXISTS prices AS SELECT * FROM ({src_sql}) LIMIT 0")
    con.execute(f"""
        MERGE INTO prices t
        USING (SELECT DISTINCT ON ({', '.join(key_cols)}) * FROM ({src_sql})) s
        ON {on_clause}
        WHEN MATCHED THEN UPDATE SET {update_set}
        WHEN NOT MATCHED THEN INSERT ({insert_cols_sql}) VALUES ({insert_vals_sql})
    """)
    logger.debug('Stored prices')


def _store_dividends(con: duckdb.DuckDBPyConnection) -> None:
    src = processed_dir / 'prices_dividends.csv'
    if not src.exists():
        return
    con.execute(f"CREATE TABLE IF NOT EXISTS dividends AS SELECT * FROM read_csv_auto('{src}') LIMIT 0")
    con.execute(f"""
        DELETE FROM dividends
        WHERE (SrcId, Date) IN (SELECT DISTINCT SrcId, Date FROM read_csv_auto('{src}'))
    """)
    con.execute(f"INSERT INTO dividends SELECT * FROM read_csv_auto('{src}')")
    logger.debug('Stored dividends')


def _store_companies(con: duckdb.DuckDBPyConnection) -> None:
    src = processed_dir / 'companies.csv'
    if not src.exists():
        return
    con.execute(f"CREATE TABLE IF NOT EXISTS companies AS SELECT * FROM read_csv_auto('{src}') LIMIT 0")
    con.execute(f"DELETE FROM companies WHERE SrcId IN (SELECT DISTINCT SrcId FROM read_csv_auto('{src}'))")
    con.execute(f"INSERT INTO companies SELECT * FROM read_csv_auto('{src}')")
    logger.debug('Stored companies')


class SimFinSource:
    def fetch_bulk(self) -> None:
        sf.set_api_key(simfin_cfg.api_key)
        sf.set_data_dir(str(raw_dir))

        before = {f: f.stat().st_mtime for f in raw_dir.glob('*.csv')}

        for dataset in BULK_DATASETS:
            logger.debug(f'Fetching {dataset.name}/{dataset.variant}/{dataset.market}')
            try:
                sf.load(
                    dataset=dataset.name,
                    variant=dataset.variant,
                    market=dataset.market,
                    refresh_days=dataset.refresh_days,
                )

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    logger.error('Rate limit exceeded. Stopping bulk fetch.')
                else:
                    logger.error(
                        f'Error fetching {dataset.name}/{dataset.variant}/{dataset.market}: {e}'
                    )
                break

        after = {f: f.stat().st_mtime for f in raw_dir.glob('*.csv')}
        if after != before:
            (raw_dir / '.fetched').touch()
            logger.debug('SimFin data updated, touching .fetched marker.')
        else:
            logger.info('fetch: no new data downloaded, skipping marker update')

    def update(self): ...

    def transform(self, feed: Literal['bulk', 'update']) -> None:
        marker = raw_dir / f'.transformed_{feed}'
        upstream = raw_dir / '.fetched'
        if is_fresh(marker, upstream):
            logger.info(f'transform({feed}): already up to date, skipping')
            return
        conn = duckdb.connect()
        _transform_fundamentals(conn)
        _transform_prices(conn)
        _transform_companies(conn)
        marker.touch()

    def store(self, feed: Literal['bulk', 'update']) -> None:
        marker = raw_dir / f'.stored_{feed}'
        upstream = raw_dir / f'.transformed_{feed}'
        if is_fresh(marker, upstream):
            logger.info(f'store({feed}): already up to date, skipping')
            return
        with duckdb.connect(config.database.path) as con:
            _store_fundamentals(con)
            _store_prices_simfin(con)
            _store_dividends(con)
            _store_companies(con)
        marker.touch()
        logger.debug('SimFin %s data stored.', feed)

    def cleanup(self) -> None:
        processed_targets = [
            processed_dir / 'fundamentals_income.csv',
            processed_dir / 'fundamentals_balance.csv',
            processed_dir / 'fundamentals_cashflow.csv',
            processed_dir / 'prices.csv',
            processed_dir / 'prices_dividends.csv',
            processed_dir / 'companies.csv',
        ]
        for path in processed_targets:
            if path.exists():
                path.unlink()
                logger.debug(f'Deleted {path}')
        for path in raw_dir.glob('*.csv'):
            path.unlink()
            logger.debug(f'Deleted {path}')


def main():

    source = SimFinSource()
    source.fetch_bulk()


if __name__ == '__main__':
    # main()
    ...
