"""`python -m dataload` — standalone ingestion CLI.

Examples::

    python -m dataload --config dataload.toml --seed-universe --refresh-universe
    python -m dataload --config dataload.toml --providers stooq,yahoo --datasets prices
    python -m dataload --config dataload.toml --providers yahoo --incremental
"""
import argparse
import logging
from collections.abc import Sequence

from dataload import make_provider, run
from dataload import universe as ul
from dataload.config import load_context

logger = logging.getLogger('dataload')


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog='dataload', description='Run market-data ingestion providers.')
    ap.add_argument('--config', required=True, help='path to a dataload TOML config')
    ap.add_argument('--providers', default='', help='comma-separated: stooq,yahoo,simfin')
    ap.add_argument('--datasets', default='', help='comma-separated; default = everything a provider supports')
    ap.add_argument('--incremental', action='store_true', help='incremental fetch where the provider supports it')
    ap.add_argument('--cleanup', action='store_true', help='delete intermediate files after load')
    ap.add_argument('--seed-universe', action='store_true', help='rebuild universe.csv from Stooq markets.csv')
    ap.add_argument('--refresh-universe', action='store_true', help='load universe.csv into the DB table')
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    ctx = load_context(args.config)

    if args.seed_universe:
        logger.info('seed-universe: %s tickers', f'{ul.seed(ctx):,}')
    if args.refresh_universe:
        logger.info('universe: %s tickers', f'{ul.refresh(ctx):,}')

    names = [n for n in args.providers.split(',') if n]
    if names:
        datasets = [d for d in args.datasets.split(',') if d] or None
        summary = run(
            ctx, [make_provider(n) for n in names],
            datasets=datasets, incremental=args.incremental, cleanup=args.cleanup,
        )
        for provider, loaded in summary.items():
            for dataset, n in loaded.items():
                logger.info('%s/%s: %s rows', provider, dataset, f'{n:,}')


if __name__ == '__main__':
    main()
