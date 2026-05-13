import logging

logging.getLogger('urllib3').setLevel(logging.INFO)
logging.getLogger('requests').setLevel(logging.INFO)

_COLORS = {
    logging.DEBUG: '\033[37m',  # white
    logging.INFO: '\033[32m',  # green
    logging.WARNING: '\033[33m',  # yellow
    logging.ERROR: '\033[31m',  # red
}
_RESET = '\033[0m'


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelno, _RESET)
        return f'{color}{super().format(record)}{_RESET}'


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        formatter = _ColorFormatter(
            fmt='%(asctime)s|%(levelname)s|%(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
