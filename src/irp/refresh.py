"""One-command refresh of the derived data chain + a staleness report.

After new raw data lands in DuckDB (via the `/ingest` ETL), two derived layers must be
rebuilt before the UI reflects it: the parquet **panels** (`irp.panel.build`) and the
**factor-cache** snapshots (`irp.factors.cache.snapshot.precompute_all`). This module runs
both in order and reports what is stale.

    uv run python -m irp.refresh            # rebuild panels + precompute the factor cache
    uv run python -m irp.refresh --status   # just print the staleness report
    uv run python -m irp.refresh --no-panels --start-year 2015 --variants A Q

Pure staleness logic (`_staleness`) is unit-tested; the heavy `refresh()` delegates to the
existing build/precompute functions.
"""
import argparse
import datetime
import logging

logger = logging.getLogger(__name__)


def _db_latest_price_date() -> datetime.date | None:
    """Most recent price Date in DuckDB (the panel's source of truth)."""
    from irp.query._common import db
    try:
        row = db().execute("SELECT max(Date) FROM prices WHERE Src = 'yahoo'").fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    return datetime.date.fromisoformat(str(row[0])[:10])


def _panel_latest_date() -> datetime.date | None:
    """Most recent date materialized in the price panel parquet."""
    try:
        from irp.panel.load import load_prices_wide
        dates = load_prices_wide('Close').dates
    except Exception:
        return None
    if dates is None or len(dates) == 0:
        return None
    import pandas as pd
    return pd.Timestamp(max(dates)).date()


def _staleness(db_date: datetime.date | None, panel_date: datetime.date | None,
               cache_dates: dict[str, datetime.date | None]) -> dict:
    """Pure staleness verdict: panel stale if DB is ahead of it; per-variant cache stale if
    the panel is ahead of (or the cache is missing) that variant."""
    panel_stale = bool(db_date and panel_date and db_date > panel_date)
    cache_stale = {v: bool(panel_date and (d is None or panel_date > d))
                   for v, d in cache_dates.items()}
    return {'panel_stale': panel_stale, 'cache_stale': cache_stale}


def status(variants=('A', 'Q')) -> dict:
    """Staleness report across DuckDB → panel → factor cache."""
    from irp.factors.cache import snapshot
    db_date = _db_latest_price_date()
    panel_date = _panel_latest_date()
    cache_dates = {v: snapshot.latest_cached(v) for v in variants}
    s = _staleness(db_date, panel_date, cache_dates)
    return {'db_latest': db_date, 'panel_latest': panel_date,
            'cache_latest': cache_dates, **s}


def refresh(rebuild_panels: bool = True, precompute: bool = True,
            start_year: int | None = None, end_year: int | None = None,
            variants=('A',), freq: str = 'QE', force: bool = False) -> dict:
    """Rebuild the derived chain: parquet panels, then the factor-cache snapshots.

    Year range defaults to the last 3 calendar years through today when omitted."""
    summary: dict = {'panels': None, 'snapshots': None}
    if rebuild_panels:
        from irp.panel.build import build_panels
        logger.info('building panels…')
        summary['panels'] = len(build_panels())
        logger.info('panels built: %s files', summary['panels'])
    if precompute:
        from irp.factors.cache import snapshot
        today = datetime.date.today()
        sy = start_year or today.year - 3
        ey = end_year or today.year
        logger.info('precomputing factor cache %s–%s variants=%s…', sy, ey, list(variants))
        end_date = min(datetime.date(ey, 12, 31), today)
        summary['snapshots'] = snapshot.precompute_all(
            datetime.date(sy, 1, 1), end_date,
            variants=list(variants), freq=freq, force=force)
        logger.info('snapshots written: %s', summary['snapshots'])
    return summary


def _print_status(st: dict) -> None:
    print('Data staleness report')
    print(f'  DuckDB latest price : {st["db_latest"]}')
    print(f'  Panel latest        : {st["panel_latest"]}   '
          f'{"STALE — rebuild panels" if st["panel_stale"] else "ok"}')
    for v, d in st['cache_latest'].items():
        flag = 'STALE — precompute' if st['cache_stale'].get(v) else 'ok'
        print(f'  Factor cache [{v}]    : {d}   {flag}')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description='Refresh derived data (panels + factor cache).')
    ap.add_argument('--status', action='store_true', help='print staleness report and exit')
    ap.add_argument('--no-panels', action='store_true', help='skip rebuilding parquet panels')
    ap.add_argument('--no-precompute', action='store_true', help='skip factor-cache precompute')
    ap.add_argument('--start-year', type=int, default=None)
    ap.add_argument('--end-year', type=int, default=None)
    ap.add_argument('--variants', nargs='+', default=['A'], choices=['A', 'Q'])
    ap.add_argument('--force', action='store_true', help='recompute cached snapshots')
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.status:
        _print_status(status())
        return
    out = refresh(rebuild_panels=not args.no_panels, precompute=not args.no_precompute,
                  start_year=args.start_year, end_year=args.end_year,
                  variants=tuple(args.variants), force=args.force)
    print(f'Refresh done: {out}')
    _print_status(status())


if __name__ == '__main__':
    main()
