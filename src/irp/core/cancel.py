import threading

_event: threading.Event = threading.Event()


def _is_cancelled() -> bool:
    return _event.is_set()


def _request_cancel() -> None:
    _event.set()


def _clear() -> None:
    _event.clear()
