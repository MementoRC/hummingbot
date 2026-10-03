import dataclasses
from decimal import Decimal
from enum import Enum
import logging
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING

from .logger import HummingbotLogger

NETWORK = DEBUG + 6


def log_encoder(obj: object) -> str | dict[str, object]:
    if isinstance(obj, Decimal):
        return str(obj)
    elif isinstance(obj, Enum):
        return str(obj)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)


__all__ = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NETWORK", "HummingbotLogger", "log_encoder"]
logging.setLoggerClass(HummingbotLogger)
logging.addLevelName(NETWORK, "NETWORK")
