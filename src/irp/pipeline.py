from typing import Literal, Protocol
import logging
from irp.core.logging import configure_logging

configure_logging()


class DataProvider(Protocol):
    def fetch(self): ...
    def update(self): ...
    def transform(self, feed: Literal['bulk', 'update']): ...
    def store(self, feed: Literal['bulk', 'update']): ...


def load_data(provider: DataProvider, feed: Literal['bulk', 'update']) -> None:
    logging.info(f"Loading {feed} data from {provider.__class__.__name__}")
    provider.fetch() if feed == 'bulk' else provider.update()
    provider.transform(feed)
    provider.store(feed)
    logging.info(f"Finished loading {feed} data from {provider.__class__.__name__}")    


def main():
    from irp.sources.stooq import StooqSource
    from irp.sources.simfin import SimFinSource

    stooq = StooqSource()
    simfin = SimFinSource()

    load_data(stooq, 'bulk')
    load_data(stooq, 'update')


if __name__ == '__main__':
    main()
