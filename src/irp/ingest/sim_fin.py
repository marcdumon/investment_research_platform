import datetime
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb
import requests

from irp.core.config import config
from irp.runner import Feed

logger = logging.getLogger(__name__)

root_dir = config.data.root_dir
simfin_cfg = config.providers.simfin
raw_dir = root_dir / simfin_cfg.raw_dir
processed_dir = root_dir / simfin_cfg.processed_dir
download_dir = raw_dir / 'download'

_BASE_URL = 'https://prod.simfin.com/api/bulk-download/s3'

_MARKETS: list[str] = ['us', 'de']
_STATEMENTS: list[str] = ['income', 'balance', 'cashflow']

# (zip variant suffix, Period label)
_ASREPORTED_FUNDAMENTAL_VARIANTS: list[tuple[str, str]] = [
    ('annual-full-asreported', 'A'),
    ('quarterly-full-asreported', 'Q'),
]
_RESTATED_FUNDAMENTAL_VARIANTS: list[tuple[str, str]] = [
    ('annual-full', 'A'),
    ('quarterly-full', 'Q'),
]
_ASREPORTED_DERIVED_VARIANTS: list[tuple[str, str]] = [
    ('annual-asreported', 'A'),
    ('quarterly-asreported', 'Q'),
]
_RESTATED_DERIVED_VARIANTS: list[tuple[str, str]] = [
    ('annual', 'A'),
    ('quarterly', 'Q'),
]


@dataclass(frozen=True)
class _Spec:
    dataset: str
    market: str | None
    variant: str | None
    refresh_days: int

    @property
    def filename(self) -> str:
        parts = [p for p in (self.market, self.dataset, self.variant) if p]
        return '-'.join(parts) + '.zip'

    @property
    def url(self) -> str:
        params = f'dataset={self.dataset}'
        if self.variant:
            params += f'&variant={self.variant}'
        if self.market:
            params += f'&market={self.market}'
        return f'{_BASE_URL}?{params}'


_SPECS: list[_Spec] = (
    [
        _Spec(s, m, v, simfin_cfg.refresh_days_fundamentals)
        for m in _MARKETS for s in _STATEMENTS
        for v, _ in _ASREPORTED_FUNDAMENTAL_VARIANTS + _RESTATED_FUNDAMENTAL_VARIANTS
    ]
    + [
        _Spec('derived', m, v, simfin_cfg.refresh_days_fundamentals)
        for m in _MARKETS
        for v, _ in _ASREPORTED_DERIVED_VARIANTS + _RESTATED_DERIVED_VARIANTS
    ]
    + [_Spec('companies', m, None, simfin_cfg.refresh_days_meta) for m in _MARKETS]
    + [_Spec('industries', None, None, simfin_cfg.refresh_days_meta)]
)


def _download_file(url: str, headers: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    try:
        # SimFin redirects to an S3 presigned URL; send auth only to SimFin,
        # then follow redirect to S3 without the Authorization header
        # (S3 presigned URLs reject extra Authorization headers).
        r = requests.get(url, headers=headers, allow_redirects=False, timeout=30)
        if r.status_code in (301, 302, 303, 307, 308):
            download_url = r.headers['Location']
            extra_headers: dict = {}
        else:
            r.raise_for_status()
            download_url = url
            extra_headers = headers

        downloaded = 0
        with requests.get(download_url, headers=extra_headers, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            with open(tmp, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        logger.debug('%s: %.0f%%', dest.name, downloaded / total * 100)
        tmp.rename(dest)
        logger.info('Downloaded %s (%s bytes)', dest.name, f'{downloaded:,}')
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _needs_download(dest: Path, refresh_days: int) -> bool:
    if not dest.exists():
        return True
    age = (datetime.date.today() - datetime.date.fromtimestamp(dest.stat().st_mtime)).days
    return age >= refresh_days


def _zip(market: str | None, dataset: str, variant: str | None = None) -> Path:
    parts = [p for p in (market, dataset, variant) if p]
    return download_dir / ('-'.join(parts) + '.zip')


def _extract(zip_path: Path, dest_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        z.extract(names[0], dest_dir)
        return dest_dir / names[0]


def _transform_fundamentals(
    conn: duckdb.DuckDBPyConnection,
    tmp: Path,
    variants: list[tuple[str, str]],
    table_suffix: str = '',
) -> None:
    for statement in _STATEMENTS:
        parts: list[str] = []
        for variant_suffix, period_label in variants:
            for market in _MARKETS:
                zp = _zip(market, statement, variant_suffix)
                if not zp.exists():
                    logger.warning('Missing %s, skipping', zp.name)
                    continue
                csv = _extract(zp, tmp)
                parts.append(
                    f"SELECT *, '{market}' AS Market, '{period_label}' AS Period "
                    f"FROM read_csv('{csv}', delim=';', union_by_name=true, null_padding=true)"
                )
                logger.debug('extracted %s', zp.name)

        if not parts:
            logger.warning('No data found for %s%s, skipping', statement, table_suffix)
            continue

        union_sql = ' UNION ALL BY NAME '.join(parts)
        out = processed_dir / f'{statement}{table_suffix}.parquet'
        conn.execute(f"""
            COPY (
                SELECT
                    Ticker,
                    SimFinId AS SrcId,
                    'simfin' AS Src,
                    * EXCLUDE (Ticker, SimFinId)
                FROM ({union_sql})
            ) TO '{out}' (FORMAT PARQUET)
        """)
        logger.info('Wrote %s', out)


def _transform_derived(
    conn: duckdb.DuckDBPyConnection,
    tmp: Path,
    variants: list[tuple[str, str]],
    table_suffix: str = '',
) -> None:
    parts: list[str] = []
    for variant_suffix, period_label in variants:
        for market in _MARKETS:
            zp = _zip(market, 'derived', variant_suffix)
            if not zp.exists():
                logger.warning('Missing %s, skipping', zp.name)
                continue
            csv = _extract(zp, tmp)
            parts.append(
                f"SELECT *, '{market}' AS Market, '{period_label}' AS Period "
                f"FROM read_csv('{csv}', delim=';', union_by_name=true, null_padding=true)"
            )
            logger.debug('extracted %s', zp.name)

    if not parts:
        logger.warning('No derived%s data found, skipping', table_suffix)
        return

    union_sql = ' UNION ALL BY NAME '.join(parts)
    out = processed_dir / f'derived{table_suffix}.parquet'
    conn.execute(f"""
        COPY (
            SELECT
                Ticker,
                SimFinId AS SrcId,
                'simfin' AS Src,
                * EXCLUDE (Ticker, SimFinId)
            FROM ({union_sql})
        ) TO '{out}' (FORMAT PARQUET)
    """)
    logger.info('Wrote %s', out)


def _transform_companies(conn: duckdb.DuckDBPyConnection, tmp: Path) -> None:
    industries_zip = _zip(None, 'industries')
    if not industries_zip.exists():
        logger.warning('Missing industries.zip, skipping companies transform')
        return
    industries_csv = _extract(industries_zip, tmp)

    parts: list[str] = []
    for market in _MARKETS:
        zp = _zip(market, 'companies')
        if not zp.exists():
            logger.warning('Missing %s, skipping', zp.name)
            continue
        csv = _extract(zp, tmp)
        parts.append(
            f"SELECT * FROM read_csv('{csv}', delim=';', union_by_name=true, null_padding=true, parallel=false)"
        )

    if not parts:
        return

    companies_union = ' UNION ALL BY NAME '.join(parts)
    out = processed_dir / 'companies.parquet'
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
            LEFT JOIN read_csv('{industries_csv}', delim=';') i
                ON c.IndustryId = i.IndustryId
        ) TO '{out}' (FORMAT PARQUET)
    """)
    logger.info('Wrote %s', out)


def _store_table(con: duckdb.DuckDBPyConnection, name: str, src: Path) -> None:
    con.execute(f'DROP TABLE IF EXISTS {name}')
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{src}')")
    logger.info('Stored %s', name)


class SimFinSource:
    SUPPORTED_FEEDS: frozenset[Feed] = frozenset({'bulk'})

    def __init__(self) -> None:
        from irp.core.markers import MarkerSet
        self.markers = MarkerSet(raw_dir)

    def fetch_bulk(self) -> None:
        api_key = simfin_cfg.api_key
        headers = {'Authorization': f'api-key {api_key}'}
        download_dir.mkdir(parents=True, exist_ok=True)

        updated = False
        failures: list[str] = []

        for spec in _SPECS:
            dest = download_dir / spec.filename
            if not _needs_download(dest, spec.refresh_days):
                logger.debug('Skipping %s (fresh)', spec.filename)
                continue
            logger.info('Downloading %s ...', spec.filename)
            try:
                _download_file(spec.url, headers, dest)
                updated = True
            except requests.HTTPError as e:
                logger.error('HTTP %s for %s: %s', e.response.status_code, spec.filename, e)
                failures.append(spec.filename)
            except Exception as e:
                logger.error('Failed %s: %s', spec.filename, e)
                failures.append(spec.filename)

        if failures:
            logger.warning('%d downloads failed: %s', len(failures), failures)

        if updated or not self.markers.exists('fetched'):
            self.markers.touch('fetched')
        else:
            logger.info('fetch_bulk: all files fresh, no downloads')

    def update(self) -> None:
        raise NotImplementedError('SimFin does not support incremental update')

    def transform(self, feed: Literal['bulk', 'update']) -> None:
        if self.markers.is_fresh('transformed_bulk', 'fetched'):
            logger.info('transform(bulk): already up to date, skipping')
            return
        processed_dir.mkdir(parents=True, exist_ok=True)
        tmp = processed_dir / '_extract_tmp'
        tmp.mkdir(exist_ok=True)
        try:
            conn = duckdb.connect()
            _transform_fundamentals(conn, tmp, _ASREPORTED_FUNDAMENTAL_VARIANTS, '')
            _transform_fundamentals(conn, tmp, _RESTATED_FUNDAMENTAL_VARIANTS, '_restated')
            _transform_derived(conn, tmp, _ASREPORTED_DERIVED_VARIANTS, '')
            _transform_derived(conn, tmp, _RESTATED_DERIVED_VARIANTS, '_restated')
            _transform_companies(conn, tmp)
            conn.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.markers.touch('transformed_bulk')

    def store(self, feed: Literal['bulk', 'update']) -> None:
        if self.markers.is_fresh('stored_bulk', 'transformed_bulk'):
            logger.info('store(bulk): already up to date, skipping')
            return
        from irp.core.db import _write_session as write_session
        with write_session() as con:
            for suffix in ('', '_restated'):
                for statement in _STATEMENTS:
                    src = processed_dir / f'{statement}{suffix}.parquet'
                    if src.exists():
                        _store_table(con, f'{statement}{suffix}', src)
                derived_src = processed_dir / f'derived{suffix}.parquet'
                if derived_src.exists():
                    _store_table(con, f'derived{suffix}', derived_src)
            companies_src = processed_dir / 'companies.parquet'
            if companies_src.exists():
                _store_table(con, 'companies', companies_src)
        self.markers.touch('stored_bulk')
        logger.info('SimFin bulk data stored')

    def cleanup(self) -> None:
        names = (
            [s for s in _STATEMENTS]
            + [f'{s}_restated' for s in _STATEMENTS]
            + ['derived', 'derived_restated', 'companies']
        )
        for name in names:
            p = processed_dir / f'{name}.parquet'
            if p.exists():
                p.unlink()
                logger.debug('Deleted %s', p)
