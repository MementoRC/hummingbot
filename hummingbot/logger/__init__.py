import dataclasses
from decimal import Decimal
from enum import Enum
import logging
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING
import sys as _sys

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


# --- hb-logger sub-package compatibility shim (MementoRC/hb-logger#9) ---
# Make `logger` and `logger.logger` resolve to THIS module tree so that the
# class `hummingbot.logger.logger.HummingbotLogger` and `logger.logger.HummingbotLogger`
# are the SAME class object at runtime — fixing isinstance failures across the
# editable-install boundary. setdefault preserves any earlier import of the
# real sub-package if it loaded first.
_sys.modules.setdefault("logger", _sys.modules[__name__])
_sys.modules.setdefault("logger.logger", _sys.modules[__name__ + ".logger"])
del _sys
