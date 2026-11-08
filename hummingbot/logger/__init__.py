"""Compat shim: hummingbot.logger -> logger (hb-logger).

The superseded submodules are aliased to the package's modules so only one
HummingbotLogger / StructLogger / ApplicationWarning / CLIHandler class exists.
The legacy sibling files stay in place at upstream content. hb-logger >= 1.0 no
longer changes global state on import, so this shim calls ``logger.install()``
once (setLoggerClass(HummingbotLogger) and the NETWORK level name, as legacy did).
"""

import sys

import logger as _pkg
import logger.application_warning as _application_warning
import logger.cli_handler as _cli_handler
import logger.logger as _logger
import logger.struct_logger as _struct_logger
from logger import CRITICAL, DEBUG, ERROR, INFO, NETWORK, WARNING, HummingbotLogger, log_encoder

# Older hb-logger registered these on import and has no install().
_install = getattr(_pkg, "install", None)
if _install is not None:
    _install(HummingbotLogger, reclass_prefixes=("hummingbot",))

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
