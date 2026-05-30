import logging

from irp.core.config import config

logging.getLogger('urllib3').setLevel(logging.INFO)
logging.getLogger('requests').setLevel(logging.INFO)
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('peewee').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

LOG_FMT = '%(asctime)s|%(levelname)s|%(name)s: %(message)s'
LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'

LEVEL_COLORS_ANSI = {
    logging.DEBUG: '\033[37m',
    logging.INFO: '\033[32m',
    logging.WARNING: '\033[33m',
    logging.ERROR: '\033[31m',
    logging.CRITICAL: '\033[91m',
}

LEVEL_COLORS_HEX = {
    logging.DEBUG: '#888888',
    logging.INFO: '#4ec94e',
    logging.WARNING: '#e0c040',
    logging.ERROR: '#e05050',
    logging.CRITICAL: '#ff4444',
}
_RESET = '\033[0m'


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = LEVEL_COLORS_ANSI.get(record.levelno, _RESET)
        return f'{color}{super().format(record)}{_RESET}'


def _configure_logging(level: int | str = config.log_level) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(_ColorFormatter(fmt=LOG_FMT, datefmt=LOG_DATEFMT))
    root.addHandler(handler)
