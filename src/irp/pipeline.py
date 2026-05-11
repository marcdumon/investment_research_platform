from typing import Literal, Protocol


class DataProvider(Protocol):
    def fetch(self): ...
    def update(self): ...
    def transform(self, feed:Literal['bulk', 'update']): ...
    def store(self, feed:Literal['bulk', 'update']): ...


def run_bulk_historicals(provider: DataProvider) -> None:
    provider.fetch()
    provider.transform('bulk')
    provider.store('bulk')


def run_updates(provider: DataProvider) -> None:
    provider.update()
    provider.transform('update')
    provider.store('update')


def main():
    from irp.sources.stooq.provider import StooqProvider
    from irp.sources.simfin.provider import SimFinProvider

    stooq = StooqProvider()
    simfin = SimFinProvider()

    run_bulk_historicals(stooq)
    run_updates(stooq)


if __name__ == '__main__':
    main()
