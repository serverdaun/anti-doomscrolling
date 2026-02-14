import logging

_FORMAT = "%(asctime)s.%(msecs)03d │ %(levelname)-5s │ %(name)s │ %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_NOISY_LOGGERS = ("mediapipe", "ultralytics", "torch")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a compact terminal format."""
    logging.basicConfig(format=_FORMAT, datefmt=_DATE_FORMAT, level=level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
