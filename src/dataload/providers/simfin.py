"""SimFin provider — produces fundamentals + company metadata (all replace-mode).

Acquire pulls bulk zips from SimFin's REST API (auth -> S3 redirect), refreshed
by file age. Normalize extracts the zips and unions them by name into one
parquet per table; the generic loader then full-replaces each table. SimFin has
no incremental feed, so no dataset declares the incremental capability.
"""
import datetime
import logging
import shutil
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import requests

from dataload.context import IngestContext
from dataload.providers.base import Capability

logger = logging.getLogger(__name__)

_BASE_URL = 'https://prod.simfin.com/api/bulk-download/s3'
_MARKETS = ['us', 'de']
_STATEMENTS = ['income', 'balance', 'cashflow']

# (zip variant suffix, Period label)
_ASREPORTED_FUNDAMENTAL_VARIANTS = [('annual-full-asreported', 'A'), ('quarterly-full-asreported', 'Q')]
_RESTATED_FUNDAMENTAL_VARIANTS = [('annual-full', 'A'), ('quarterly-full', 'Q')]
_ASREPORTED_DERIVED_VARIANTS = [('annual-asreported', 'A'), ('quarterly-asreported', 'Q')]
_RESTATED_DERIVED_VARIANTS = [('annual', 'A'), ('quarterly', 'Q')]


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


def _specs(refresh_fundamentals: int, refresh_meta: int) -> list[_Spec]:
    return (
        [_Spec(s, m, v, refresh_fundamentals)
         for m in _MARKETS for s in _STATEMENTS
         for v, _ in _ASREPORTED_FUNDAMENTAL_VARIANTS + _RESTATED_FUNDAMENTAL_VARIANTS]
        + [_Spec('derived', m, v, refresh_fundamentals)
           for m in _MARKETS
           for v, _ in _ASREPORTED_DERIVED_VARIANTS + _RESTATED_DERIVED_VARIANTS]
        + [_Spec('companies', m, None, refresh_meta) for m in _MARKETS]
        + [_Spec('industries', None, None, refresh_meta)]
    )


def _needs_download(dest: Path, refresh_days: int) -> bool:
    if not dest.exists():
        return True
    age = (datetime.date.today() - datetime.date.fromtimestamp(dest.stat().st_mtime)).days
    return age >= refresh_days


def _download_file(url: str, headers: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    try:
        # SimFin redirects to a presigned S3 URL; send auth only to SimFin, then
        # follow the redirect without it (S3 rejects extra Authorization headers).
        r = requests.get(url, headers=headers, allow_redirects=False, timeout=30)
        if r.status_code in (301, 302, 303, 307, 308):
            download_url, extra_headers = r.headers['Location'], {}
        else:
            r.raise_for_status()
            download_url, extra_headers = url, headers
        with requests.get(download_url, headers=extra_headers, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(tmp, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
        tmp.rename(dest)
        logger.info('Downloaded %s', dest.name)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _zip(download_dir: Path, market: str | None, dataset: str, variant: str | None = None) -> Path:
    parts = [p for p in (market, dataset, variant) if p]
    return download_dir / ('-'.join(parts) + '.zip')


def _extract(zip_path: Path, dest_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        z.extract(name, dest_dir)
        return dest_dir / name


def _union_fundamentals(conn, download_dir: Path, tmp: Path, processed: Path,
                        variants: list[tuple[str, str]], suffix: str) -> None:
    for statement in _STATEMENTS:
        parts: list[str] = []
        for variant_suffix, period in variants:
            for market in _MARKETS:
                zp = _zip(download_dir, market, statement, variant_suffix)
                if not zp.exists():
                    logger.warning('Missing %s, skipping', zp.name)
                    continue
                csv = _extract(zp, tmp)
                parts.append(f"SELECT *, '{market}' AS Market, '{period}' AS Period "
                             f"FROM read_csv('{csv}', delim=';', union_by_name=true, null_padding=true)")
        if not parts:
            logger.warning('No data for %s%s, skipping', statement, suffix)
            continue
        out = processed / f'{statement}{suffix}.parquet'
        conn.execute(f"""
            COPY (SELECT Ticker, SimFinId AS SrcId, 'simfin' AS Src, * EXCLUDE (Ticker, SimFinId)
                  FROM ({' UNION ALL BY NAME '.join(parts)})) TO '{out}' (FORMAT PARQUET)
        """)
        logger.info('Wrote %s', out)


def _union_derived(conn, download_dir: Path, tmp: Path, processed: Path,
                   variants: list[tuple[str, str]], suffix: str) -> None:
    parts: list[str] = []
    for variant_suffix, period in variants:
        for market in _MARKETS:
            zp = _zip(download_dir, market, 'derived', variant_suffix)
            if not zp.exists():
                logger.warning('Missing %s, skipping', zp.name)
                continue
            csv = _extract(zp, tmp)
            parts.append(f"SELECT *, '{market}' AS Market, '{period}' AS Period "
                         f"FROM read_csv('{csv}', delim=';', union_by_name=true, null_padding=true)")
    if not parts:
        logger.warning('No derived%s data, skipping', suffix)
        return
    out = processed / f'derived{suffix}.parquet'
    conn.execute(f"""
        COPY (SELECT Ticker, SimFinId AS SrcId, 'simfin' AS Src, * EXCLUDE (Ticker, SimFinId)
              FROM ({' UNION ALL BY NAME '.join(parts)})) TO '{out}' (FORMAT PARQUET)
    """)
    logger.info('Wrote %s', out)


def _union_companies(conn, download_dir: Path, tmp: Path, processed: Path) -> None:
    industries_zip = _zip(download_dir, None, 'industries')
    if not industries_zip.exists():
        logger.warning('Missing industries.zip, skipping companies')
        return
    industries_csv = _extract(industries_zip, tmp)
    parts: list[str] = []
    for market in _MARKETS:
        zp = _zip(download_dir, market, 'companies')
        if not zp.exists():
            logger.warning('Missing %s, skipping', zp.name)
            continue
        csv = _extract(zp, tmp)
        parts.append(f"SELECT * FROM read_csv('{csv}', delim=';', union_by_name=true, null_padding=true, parallel=false)")
    if not parts:
        return
    out = processed / 'companies.parquet'
    conn.execute(f"""
        COPY (
            SELECT c.Ticker, c."Company Name", i.Industry, i.Sector, c."Business Summary",
                   c."End of financial year (month)", c."Number Employees", c.CIK, c.ISIN,
                   c."Main Currency", c.IndustryId, c.Market, c.SimFinId AS SrcId
            FROM ({' UNION ALL BY NAME '.join(parts)}) c
            LEFT JOIN read_csv('{industries_csv}', delim=';') i ON c.IndustryId = i.IndustryId
        ) TO '{out}' (FORMAT PARQUET)
    """)
    logger.info('Wrote %s', out)


class SimFinProvider:
    name = 'simfin'

    _DATASETS = [
        'income', 'balance', 'cashflow', 'derived',
        'income_restated', 'balance_restated', 'cashflow_restated', 'derived_restated',
        'companies',
    ]

    def capabilities(self) -> dict[str, Capability]:
        return {name: Capability(incremental=False) for name in self._DATASETS}

    def produce(self, ctx: IngestContext, datasets: Sequence[str], *, incremental: bool) -> dict[str, Path]:
        if incremental:
            logger.info('simfin: no incremental feed, skipping')
            return {}
        if not any(d in self._DATASETS for d in datasets):
            return {}
        self._acquire(ctx)
        return self._normalize(ctx, datasets)

    def _acquire(self, ctx: IngestContext) -> None:
        cfg = ctx.cfg('simfin')
        download_dir = ctx.raw_dir('simfin') / 'download'
        download_dir.mkdir(parents=True, exist_ok=True)
        headers = {'Authorization': f"api-key {cfg['api_key']}"}
        specs = _specs(cfg['refresh_days_fundamentals'], cfg['refresh_days_meta'])
        for spec in specs:
            dest = download_dir / spec.filename
            if not _needs_download(dest, spec.refresh_days):
                continue
            try:
                _download_file(spec.url, headers, dest)
            except Exception as e:  # noqa: BLE001
                logger.error('Failed %s: %s', spec.filename, e)

    def _normalize(self, ctx: IngestContext, datasets: Sequence[str]) -> dict[str, Path]:
        download_dir = ctx.raw_dir('simfin') / 'download'
        processed = ctx.processed_dir('simfin')
        processed.mkdir(parents=True, exist_ok=True)
        tmp = processed / '_extract_tmp'
        tmp.mkdir(exist_ok=True)
        conn = duckdb.connect()
        try:
            _union_fundamentals(conn, download_dir, tmp, processed, _ASREPORTED_FUNDAMENTAL_VARIANTS, '')
            _union_fundamentals(conn, download_dir, tmp, processed, _RESTATED_FUNDAMENTAL_VARIANTS, '_restated')
            _union_derived(conn, download_dir, tmp, processed, _ASREPORTED_DERIVED_VARIANTS, '')
            _union_derived(conn, download_dir, tmp, processed, _RESTATED_DERIVED_VARIANTS, '_restated')
            _union_companies(conn, download_dir, tmp, processed)
        finally:
            conn.close()
            shutil.rmtree(tmp, ignore_errors=True)
        out: dict[str, Path] = {}
        for name in datasets:
            p = processed / f'{name}.parquet'
            if name in self._DATASETS and p.exists():
                out[name] = p
        return out

    def cleanup(self, ctx: IngestContext) -> None:
        processed = ctx.processed_dir('simfin')
        for name in self._DATASETS:
            p = processed / f'{name}.parquet'
            if p.exists():
                p.unlink()
