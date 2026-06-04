"""Backward-compat shim for the canonical ``logger`` sub-package.

The canonical implementation lives in the ``logger`` package shipped by the
hb-logger sub-package (editable-installed into this monorepo). This module
preserves the historical ``hummingbot.logger`` import path so that the 100+
existing consumers do not need to be rewritten — every name re-exported here
is the SAME object as the one in the sub-package, so ``isinstance`` checks
work regardless of which dotted path the caller imported from.

See MementoRC/hb-logger#9 (dual-import isinstance failure) for the bug this
fixes, and MementoRC/hb-logger#1 (full extraction plan) for the larger context.
"""

from logger import CRITICAL, DEBUG, ERROR, INFO, NETWORK, WARNING, HummingbotLogger, log_encoder  # noqa: F401

__all__ = [
    "CRITICAL",
    "DEBUG",
    "ERROR",
    "HummingbotLogger",
    "INFO",
    "NETWORK",
    "WARNING",
    "log_encoder",
]
