"""Compat shim: hummingbot.logger -> logger (hb-logger).

The superseded submodules are aliased to the package's modules so only one
HummingbotLogger / StructLogger / ApplicationWarning / CLIHandler class exists.
The legacy sibling files stay in place at upstream content. Importing ``logger``
already runs setLoggerClass(HummingbotLogger) and addLevelName(NETWORK) once.
"""

import sys

import logger.application_warning as _application_warning
import logger.cli_handler as _cli_handler
import logger.logger as _logger
import logger.struct_logger as _struct_logger
from logger import CRITICAL, DEBUG, ERROR, INFO, NETWORK, WARNING, HummingbotLogger, log_encoder

for _name, _mod in (
    ("application_warning", _application_warning),
    ("cli_handler", _cli_handler),
    ("logger", _logger),
    ("struct_logger", _struct_logger),
):
    sys.modules[f"{__name__}.{_name}"] = _mod
    globals()[_name] = _mod

__all__ = [
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "NETWORK",
    "HummingbotLogger",
    "log_encoder",
]
