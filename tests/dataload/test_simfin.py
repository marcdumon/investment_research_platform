"""SimFin provider: REST acquire + union normalize -> replace-mode tables."""
import zipfile
from contextlib import contextmanager

import duckdb
import pandas as pd

from dataload.context import IngestContext
from dataload.load import load_dataset
from dataload.providers.simfin import SimFinProvider, _needs_download, _Spec


def _ctx(tmp_path) -> IngestContext:
    @contextmanager
    def connect():
        con = duckdb.connect(str(tmp_path / 'db.duckdb'))
        try:
            yield con
        finally:
            con.close()

    cfg = {'simfin': {'raw_dir': 'simfin/raw', 'processed_dir': 'simfin/processed',
                      'api_key': 'x', 'refresh_days_fundamentals': 1, 'refresh_days_meta': 1}}
    return IngestContext(tmp_path, cfg, connect)


def test_spec_filename_and_url() -> None:
    s = _Spec('income', 'us', 'annual-full-asreported', 1)
    assert s.filename == 'us-income-annual-full-asreported.zip'
    assert 'dataset=income' in s.url
    assert 'variant=annual-full-asreported' in s.url
    assert 'market=us' in s.url


def test_spec_filename_without_variant_or_market() -> None:
    assert _Spec('industries', None, None, 1).filename == 'industries.zip'


def test_needs_download_when_missing(tmp_path) -> None:
    assert _needs_download(tmp_path / 'nope.zip', 1) is True


def test_needs_download_when_fresh(tmp_path) -> None:
    p = tmp_path / 'f.zip'
    p.write_text('x')
    assert _needs_download(p, 7) is False


def test_capabilities_full_replace_no_incremental() -> None:
    caps = SimFinProvider().capabilities()
    assert {'income', 'balance', 'cashflow', 'derived', 'companies'} <= set(caps)
    assert all(not c.incremental for c in caps.values())


def test_normalize_fundamentals_from_zip_and_load(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    download = ctx.raw_dir('simfin') / 'download'
    download.mkdir(parents=True)
    with zipfile.ZipFile(download / 'us-income-annual-full-asreported.zip', 'w') as z:
        z.writestr('us-income-annual-full-asreported.csv', 'Ticker;SimFinId;Revenue\nAAPL;111;1000\n')

    out = SimFinProvider()._normalize(ctx, ['income'])
    assert 'income' in out
    df = pd.read_parquet(out['income'])
    assert {'Ticker', 'SrcId', 'Src', 'Market', 'Period'} <= set(df.columns)
    assert df.iloc[0]['Src'] == 'simfin'
    assert df.iloc[0]['SrcId'] == 111

    with ctx.connect() as con:
        assert load_dataset(con, 'income', out['income']) == 1
