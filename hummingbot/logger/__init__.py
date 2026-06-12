import dataclasses
import logging
import sys as _sys
from decimal import Decimal
from enum import Enum
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING

from .logger import HummingbotLogger

NETWORK = DEBUG + 6


def log_encoder(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    elif isinstance(obj, Enum):
        return str(obj)
    elif dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)


__all__ = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NETWORK", "HummingbotLogger", "log_encoder"]
logging.setLoggerClass(HummingbotLogger)
logging.addLevelName(NETWORK, "NETWORK")


# hb-logger sub-package compatibility shim (MementoRC/hb-logger#9).
# When the hb-logger sub-package is installed in editable mode alongside
# hummingbot, both packages import their own copy of the HummingbotLogger
# class. Python's logging.setLoggerClass() registers whichever runs last,
# and live logger instances become that class — breaking
# isinstance(obj, hummingbot.logger.logger.HummingbotLogger) checks across
# the codebase. Aliasing the sub-package's module paths to hummingbot's
# ensures both import paths resolve to the same class object.
_sys.modules.setdefault("logger", _sys.modules[__name__])
_sys.modules.setdefault("logger.logger", _sys.modules[__name__ + ".logger"])
del _sys
