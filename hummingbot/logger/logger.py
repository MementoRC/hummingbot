#!/usr/bin/env python
"""Backward-compatible re-export of ``HummingbotLogger``.

The full implementation now lives in the ``logger`` sub-package
(``sub-packages/logger``, MementoRC/hb-logger). This module used to carry a
full parallel copy of that class, including a ``notify()``/``network()``
implementation that looked up ``HummingbotApplication.main_application()``
directly — a hidden ``logger -> hummingbot_application`` import cycle. The
sub-package's ``HummingbotLogger`` replaces that lookup with an explicit
callback-registration API (``register_notify_handler`` /
``register_network_handler``), wired once at app startup in
``HummingbotApplication.__init__`` instead of being resolved lazily on every
log call.

This is now a genuine import, so ``hummingbot.logger.logger.HummingbotLogger``
and ``logger.logger.HummingbotLogger`` are the same class object.
"""

from logger.logger import HummingbotLogger

__all__ = ["HummingbotLogger"]
