"""`python -m dataload` — standalone ingestion CLI.

Two modes:
  * Interactive (default in a terminal with no provider/universe flags): questionary
    prompts pick providers, datasets, feed, and steps — same feel as the host app.
  * Flag-driven (scripts / CI / non-tty): pass flags explicitly.

Examples::

    dataload                                          # interactive prompts
    dataload --providers stooq,yahoo --datasets prices
    dataload --providers yahoo --incremental
    dataload --seed-universe --refresh-universe

`--config` defaults to ./dataload.toml (override with --config or $DATALOAD_CONFIG).
The interactive prompts need `questionary`; without it the CLI tells you to use flags.
"""
import argparse
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from dataload import make_provider, run
from dataload import universe as ul
from dataload.config import load_context
from dataload.context import IngestContext
from dataload.providers import PROVIDERS

logger = logging.getLogger('dataload')

DEFAULT_CONFIG = 'dataload.toml'


@dataclass(frozen=True, slots=True)
class Plan:
    """What the user wants run — produced by either the prompts or the flags."""
    providers: list[str]
    datasets: list[str] | None
    incremental: bool
    ingest: bool
    cleanup: bool
    seed: bool
    refresh: bool


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog='dataload', description='Run market-data ingestion providers.')
    ap.add_argument('--config', default=os.environ.get('DATALOAD_CONFIG', DEFAULT_CONFIG),
                    help='path to a dataload TOML config (default: %(default)s)')
    ap.add_argument('-i', '--interactive', action='store_true', help='pick options via interactive prompts')
    ap.add_argument('--providers', default='', help='comma-separated: ' + ','.join(PROVIDERS))
    ap.add_argument('--datasets', default='', help='comma-separated; default = everything a provider supports')
    ap.add_argument('--incremental', action='store_true', help='incremental fetch where the provider supports it')
    ap.add_argument('--cleanup', action='store_true', help='delete intermediate files after load')
    ap.add_argument('--seed-universe', action='store_true', help='rebuild universe.csv from Stooq markets.csv')
    ap.add_argument('--refresh-universe', action='store_true', help='load universe.csv into the DB table')
    return ap.parse_args(argv)


def _plan_from_flags(args: argparse.Namespace) -> Plan:
    return Plan(
        providers=[n for n in args.providers.split(',') if n],
        datasets=[d for d in args.datasets.split(',') if d] or None,
        incremental=args.incremental,
        ingest=bool(args.providers),
        cleanup=args.cleanup,
        seed=args.seed_universe,
        refresh=args.refresh_universe,
    )


def _prompt() -> Plan | None:
    """Interactive plan via questionary; None if the user aborts. Lazy-imports questionary."""
    try:
        import questionary
        from questionary import Choice
    except ImportError:
        logger.error('Interactive mode needs questionary (pip install questionary), or pass flags instead.')
        return None

    providers = questionary.checkbox(
        'Providers:', choices=[Choice(n, checked=n != 'simfin') for n in PROVIDERS],
    ).ask()
    if not providers:
        return None

    caps: dict[str, object] = {}
    for n in providers:
        caps.update(make_provider(n).capabilities())
    datasets = questionary.checkbox(
        'Datasets:', choices=[Choice(d, checked=True) for d in caps],
    ).ask()
    if datasets is None:
        return None

    feed = questionary.select(
        'Feed:', choices=[Choice('bulk    (full history)', 'bulk'), Choice('update  (incremental)', 'update')],
    ).ask()
    if feed is None:
        return None

    steps = questionary.checkbox(
        'Steps:', choices=[
            Choice('ingest   (fetch + normalize + load)', 'ingest', checked=True),
            Choice('cleanup', 'cleanup'),
            Choice('seed-universe', 'seed-universe'),
            Choice('refresh-universe', 'refresh-universe'),
        ],
    ).ask()
    if steps is None:
        return None

    return Plan(
        providers=providers,
        datasets=datasets or None,
        incremental=feed == 'update',
        ingest='ingest' in steps,
        cleanup='cleanup' in steps,
        seed='seed-universe' in steps,
        refresh='refresh-universe' in steps,
    )


def _execute(ctx: IngestContext, plan: Plan) -> None:
    if plan.seed:
        logger.info('seed-universe: %s tickers', f'{ul.seed(ctx):,}')
    if plan.refresh:
        logger.info('universe: %s tickers', f'{ul.refresh(ctx):,}')

    if not plan.providers:
        return
    providers = [make_provider(n) for n in plan.providers]
    if plan.ingest:
        summary = run(ctx, providers, datasets=plan.datasets, incremental=plan.incremental, cleanup=plan.cleanup)
        for name, loaded in summary.items():
            for dataset, n in loaded.items():
                logger.info('%s/%s: %s rows', name, dataset, f'{n:,}')
    elif plan.cleanup:
        for provider in providers:
            provider.cleanup(ctx)
            logger.info('%s: cleaned', provider.name)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    flag_driven = bool(args.providers or args.seed_universe or args.refresh_universe)
    interactive = args.interactive or (not flag_driven and sys.stdin.isatty())

    ctx = load_context(args.config)
    plan = _prompt() if interactive else _plan_from_flags(args)
    if plan is None:
        return
    _execute(ctx, plan)


if __name__ == '__main__':
    main()
