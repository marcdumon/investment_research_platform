import logging

_COLORS = {
    logging.DEBUG:   "\033[37m",   # white
    logging.INFO:    "\033[32m",   # green
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR:   "\033[31m",   # red
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelno, _RESET)
        return f"{color}{super().format(record)}{_RESET}"


def configure_logging(level: int = logging.DEBUG) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
        formatter = _ColorFormatter(fmt)
        handler.setFormatter(formatter)
        root.addHandler(handler)


