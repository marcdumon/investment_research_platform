import logging
from collections import deque

_log_buffer: deque[tuple[int, str]] = deque(maxlen=500)
_run_active: bool = False
_run_done: bool = False


class DequeHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _log_buffer.append((record.levelno, self.format(record)))
