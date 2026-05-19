import threading

_event: threading.Event = threading.Event()


def is_cancelled() -> bool:
    return _event.is_set()


def request_cancel() -> None:
    _event.set()


def clear() -> None:
    _event.clear()
