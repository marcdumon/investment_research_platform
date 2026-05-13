from collections import defaultdict
import logging
from dataclasses import dataclass
from typing import Literal, Sequence

import duckdb
import requests
import simfin as sf

from irp.core.config import config

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
        conn.execute(f"COPY ({union_sql}) TO '{out}' (FORMAT CSV, HEADER)")
        logger.debug('Wrote %s', out)


def _transform_shareprices(conn: duckdb.DuckDBPyConnection) -> None:
    shareprices_files = sorted(raw_dir.glob('*-shareprices-*.csv'))
    if not shareprices_files:
        return

    src = ' UNION ALL '.join(
        f"SELECT * FROM read_csv('{f}', delim=';')" for f in shareprices_files
    )

    prices_out = processed_dir / 'shareprices.csv'
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

    divs_out = processed_dir / 'shareprices_dividends.csv'
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


class SimFinSource:
    def fetch_bulk(self) -> None:
        print(simfin_cfg.api_key)
        sf.set_api_key(simfin_cfg.api_key)
        sf.set_data_dir(str(raw_dir))

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

    def update(self): ...

    def transform(self, feed: Literal['bulk', 'update']) -> None:
        conn = duckdb.connect()
        _transform_fundamentals(conn)
        _transform_shareprices(conn)
        _transform_companies(conn)

    def store(self, feed: Literal['bulk', 'update']) -> None: ...

    def cleanup(self) -> None: ...


def main():

    source = SimFinSource()
    source.fetch_bulk()


if __name__ == '__main__':
    # main()
    ...
