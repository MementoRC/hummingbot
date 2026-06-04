"""Backward-compat shim for the canonical ``logger.logger`` sub-package module.

The canonical implementation lives in the hb-logger sub-package at
``logger/logger.py``. This module preserves the historical
``hummingbot.logger.logger`` import path so that callers doing
``from hummingbot.logger.logger import HummingbotLogger`` get the SAME class
object as callers doing ``from logger.logger import HummingbotLogger`` — making
``isinstance`` checks work across both paths.

See MementoRC/hb-logger#9 (dual-import isinstance failure).
"""

from logger.logger import TESTING_TOOLS, HummingbotLogger, _srcfile, currentframe  # noqa: F401
