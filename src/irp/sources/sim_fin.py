from collections import defaultdict
import logging
import shutil
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
Market = Literal['us', 'de', 'ca', 'cn']


@dataclass(frozen=True)
class FundamentalsDataset:
    name: Literal['income', 'balance', 'cashflow']
    variant: FundamentalsVariant
    market: Market
    refresh_days: int = simfin_cfg.refresh_days_fundamentals


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


SimFinDataset = FundamentalsDataset | CompaniesDataset | MetaDataset


_FUNDAMENTALS_NAMES: list[FundamentalsNames] = ['income', 'balance', 'cashflow']
_FUNDAMENTALS_VARIANTS: list[FundamentalsVariant] = ['annual', 'quarterly']
_MARKETS: list[Market] = ['us', 'de']
_META_NAMES: list[Literal['markets', 'industries']] = ['industries']

BULK_DATASETS: Sequence[SimFinDataset] = (
    [
        FundamentalsDataset(name, variant, market)
        for name in _FUNDAMENTALS_NAMES
        for variant in _FUNDAMENTALS_VARIANTS
        for market in _MARKETS
    ]
    + [CompaniesDataset(market) for market in _MARKETS]
    + [MetaDataset(name) for name in _META_NAMES]
)


def _configure_sf() -> None:
    sf.set_api_key(simfin_cfg.api_key)
    sf.set_data_dir(str(raw_dir))


def _download_dataset(dataset: SimFinDataset) -> bool:
    try:
        sf.load(
            dataset=dataset.name,
            variant=dataset.variant,
            market=dataset.market,
            refresh_days=dataset.refresh_days,
        )
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.error('Rate limit exceeded. Stopping bulk fetch.')
        else:
            logger.error(
                f'Error fetching {dataset.name}/{dataset.variant}/{dataset.market}: {e}'
            )
        return False


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
        out = processed_dir / f'{statement}.csv'
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
        src = processed_dir / f'{statement}.csv'
        if not src.exists():
            continue
        table = statement
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM read_csv_auto('{src}') LIMIT 0"
        )
        con.execute(f"""
            MERGE INTO {table} t
            USING (
                SELECT DISTINCT ON (SrcId, "Fiscal Year", "Fiscal Period", Period) *
                FROM read_csv_auto('{src}')
            ) s
            ON t.SrcId = s.SrcId
            AND t."Fiscal Year" = s."Fiscal Year"
            AND t."Fiscal Period" = s."Fiscal Period"
            AND t.Period = s.Period
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)
        logger.debug('Stored %s', table)


def _store_companies(con: duckdb.DuckDBPyConnection) -> None:
    src = processed_dir / 'companies.csv'
    if not src.exists():
        return
    con.execute(
        f"CREATE TABLE IF NOT EXISTS companies AS SELECT * FROM read_csv_auto('{src}') LIMIT 0"
    )
    con.execute(
        f"DELETE FROM companies WHERE SrcId IN (SELECT DISTINCT SrcId FROM read_csv_auto('{src}'))"
    )
    con.execute(f"INSERT INTO companies SELECT * FROM read_csv_auto('{src}')")
    logger.debug('Stored companies')


class SimFinSource:
    def fetch_bulk(self) -> None:
        _configure_sf()
        before = {f: f.stat().st_mtime for f in raw_dir.glob('*.csv')}

        for dataset in BULK_DATASETS:
            logger.debug(f'Fetching {dataset.name}/{dataset.variant}/{dataset.market}')
            success = _download_dataset(dataset)
            if not success:
                break

        marker = raw_dir / '.fetched'
        after = {f: f.stat().st_mtime for f in raw_dir.glob('*.csv')}
        if after != before:
            marker.touch()
            logger.debug('SimFin data updated, touching .fetched marker.')
        elif not marker.exists():
            marker.touch()
            logger.debug('SimFin data already on disk, creating baseline .fetched marker.')
        else:
            logger.info('fetch: no new data downloaded, skipping marker update')

    def update(self) -> None:
        """SimFin has no incremental update path without shareprices. No-op."""
        logger.info('SimFin update: nothing to do (shareprices removed)')

    def transform(self, feed: Literal['bulk', 'update']) -> None:
        if feed == 'update':
            logger.info('transform(update): nothing to do')
            return
        marker = raw_dir / '.transformed_bulk'
        upstream = raw_dir / '.fetched'
        if is_fresh(marker, upstream):
            logger.info('transform(bulk): already up to date, skipping')
            return
        conn = duckdb.connect()
        _transform_fundamentals(conn)
        _transform_companies(conn)
        marker.touch()

    def store(self, feed: Literal['bulk', 'update']) -> None:
        if feed == 'update':
            logger.info('store(update): nothing to do')
            return
        marker = raw_dir / '.stored_bulk'
        upstream = raw_dir / '.transformed_bulk'
        if is_fresh(marker, upstream):
            logger.info('store(bulk): already up to date, skipping')
            return
        with duckdb.connect(config.database.path) as con:
            _store_fundamentals(con)
            _store_companies(con)
        marker.touch()
        logger.debug('SimFin bulk data stored.')

    def cleanup(self) -> None:
        processed_targets = [
            processed_dir / 'income.csv',
            processed_dir / 'balance.csv',
            processed_dir / 'cashflow.csv',
            processed_dir / 'companies.csv',
        ]
        for path in processed_targets:
            if path.exists():
                path.unlink()
                logger.debug(f'Deleted {path}')
        for d in (raw_dir / 'info', raw_dir / 'cache'):
            if d.exists():
                shutil.rmtree(d)
                logger.debug(f'Deleted {d}')
