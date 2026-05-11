
from typing import Protocol


class DataProvider(Protocol):
    def fetch(self): ...
    def update(self): ...  
    def transform(self, raw): ...
    def store(self, data): ...


def run_bulk_historicals(provider: DataProvider) -> None:
    raw = provider.fetch()
    clean = provider.transform(raw)
    provider.store(clean)


def run_updates(provider: DataProvider) -> None:
    raw = provider.update()
    clean = provider.transform(raw)
    provider.store(clean)


def main():
    from irp.sources.stooq.provider import StooqProvider
    from irp.sources.simfin.provider import SimFinProvider

    stooq = StooqProvider()
    simfin = SimFinProvider()

    run_historicals(stooq)
    run_updates(simfin)


if __name__ == "__main__":
    main()
