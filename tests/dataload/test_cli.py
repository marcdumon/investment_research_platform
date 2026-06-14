"""Standalone CLI: TOML config loader + `python -m dataload` entry point."""
import duckdb
import pandas as pd

from dataload.__main__ import main
from dataload.config import load_context


def _write_config(tmp_path) -> object:
    cfg = tmp_path / 'dataload.toml'
    cfg.write_text(
        f'[database]\npath = "{tmp_path}/db.duckdb"\n'
        f'[data]\nroot_dir = "{tmp_path}"\n'
        '[providers.stooq]\nraw_dir = "stooq/raw"\nprocessed_dir = "stooq/processed"\n'
        'bulk_files = ["d_us_txt.zip"]\nupdate_file = "data_d.txt"\n'
    )
    return cfg


def test_load_context_maps_toml(tmp_path) -> None:
    ctx = load_context(_write_config(tmp_path))
    assert ctx.data_root == tmp_path
    assert ctx.cfg('stooq')['bulk_files'] == ['d_us_txt.zip']
    assert ctx.raw_dir('stooq') == tmp_path / 'stooq' / 'raw'


def test_load_context_connect_persists_writes(tmp_path) -> None:
    ctx = load_context(_write_config(tmp_path))
    with ctx.connect() as con:
        con.execute('CREATE TABLE t (x INTEGER)')
        con.execute('INSERT INTO t VALUES (1)')
    with duckdb.connect(f'{tmp_path}/db.duckdb', read_only=True) as c2:
        assert c2.execute('SELECT x FROM t').fetchone()[0] == 1


def test_cli_seed_and_refresh_universe(tmp_path) -> None:
    raw = tmp_path / 'stooq' / 'raw'
    raw.mkdir(parents=True)
    pd.DataFrame([['t.v', 'cryptocurrencies'], ['t.us', 'nyse stocks']],
                 columns=['ticker', 'market']).to_csv(raw / 'markets.csv', index=False)
    cfg = _write_config(tmp_path)

    main(['--config', str(cfg), '--seed-universe', '--refresh-universe'])

    uni = pd.read_csv(tmp_path / 'universe.csv')
    assert uni.query("Ticker == 'T'").iloc[0]['Market'] == 'nyse stocks'  # the dedup fix, via CLI
    with duckdb.connect(f'{tmp_path}/db.duckdb', read_only=True) as c:
        assert c.execute("SELECT Market FROM universe WHERE Ticker = 'T'").fetchone()[0] == 'nyse stocks'
