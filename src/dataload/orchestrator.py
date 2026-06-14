"""Run providers through the generic load, with capability-based dataset gating."""
import logging
from collections.abc import Sequence

from dataload.context import IngestContext
from dataload.load import load_dataset
from dataload.providers.base import Provider

logger = logging.getLogger(__name__)


def run(
    ctx: IngestContext,
    providers: Sequence[Provider],
    datasets: Sequence[str] | None = None,
    *,
    incremental: bool = False,
    cleanup: bool = False,
) -> dict[str, dict[str, int]]:
    """Produce + load the requested datasets from each provider.

    `datasets=None` means every dataset a provider can produce. Requested names
    a provider does not support are silently skipped. With `incremental=True`,
    only datasets whose capability allows it are run. Returns
    `{provider_name: {dataset: rows_loaded}}`.
    """
    summary: dict[str, dict[str, int]] = {}
    for provider in providers:
        caps = provider.capabilities()
        requested = list(datasets) if datasets is not None else list(caps)
        wanted = [d for d in requested if d in caps]
        if incremental:
            wanted = [d for d in wanted if caps[d].incremental]

        if not wanted:
            logger.info('%s: no matching datasets, skipping', provider.name)
            summary[provider.name] = {}
            continue

        logger.info('%s: producing %s (incremental=%s)', provider.name, wanted, incremental)
        produced = provider.produce(ctx, wanted, incremental=incremental)

        loaded: dict[str, int] = {}
        if produced:
            with ctx.connect() as con:
                for dataset, parquet in produced.items():
                    loaded[dataset] = load_dataset(con, dataset, parquet)
                    logger.info('%s: loaded %s rows into %s', provider.name, loaded[dataset], dataset)
        if cleanup:
            provider.cleanup(ctx)
        summary[provider.name] = loaded
    return summary
