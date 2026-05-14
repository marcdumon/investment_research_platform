import logging
from typing import Literal, Protocol

from irp.core.logging import configure_logging

configure_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataProvider(Protocol):
    def fetch_bulk(self): ...
    def update(self): ...
    def transform(self, feed: Literal['bulk', 'update']): ...
    def store(self, feed: Literal['bulk', 'update']): ...
    def cleanup(self): ...


def load_data(provider: DataProvider, feed: Literal['bulk', 'update']) -> None:
    logger.info(f'Loading {feed} data from {provider.__class__.__name__}')
    provider.fetch_bulk() if feed == 'bulk' else provider.update()
    provider.transform(feed)
    provider.store(feed)
    logger.info(f'Finished loading {feed} data from {provider.__class__.__name__}')


def main():
    from irp.sources.sim_fin import SimFinSource
    from irp.sources.stooq import StooqSource

    stooq = StooqSource()
    simfin = SimFinSource()

    # load_data(stooq, 'bulk')
    # load_data(stooq, 'update')
    # stooq.cleanup()

    # load_data(simfin, 'bulk')
    load_data(simfin, 'update')

    # simfin.cleanup()


if __name__ == '__main__':
    main()
