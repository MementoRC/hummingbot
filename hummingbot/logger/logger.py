#!/usr/bin/env python

from __future__ import annotations

import io
from logging import Logger as PythonLogger
import os
import sys
import time
import traceback
from typing import Callable, ClassVar, Type

import pandas as pd

from .application_warning import ApplicationWarning

TESTING_TOOLS = ["unittest", "pytest"]

#  --- Copied from logging module ---
if hasattr(sys, "_getframe"):

    def currentframe():
        return sys._getframe(3)
else:  # pragma: no cover

    def currentframe():
        """Return the frame object for the caller's stack frame."""
        try:
            raise Exception
        except Exception:
            tb = sys.exc_info()[2]
            assert tb is not None
            return tb.tb_frame.f_back
#  --- Copied from logging module ---


class HummingbotLogger(PythonLogger):
    _notify_callback: ClassVar[Callable[[str], None] | None] = None
    _network_callback: ClassVar[Callable[[ApplicationWarning], None] | None] = None

    @classmethod
    def register_notify_handler(cls, handler: Callable[[str], None]) -> None:
        cls._notify_callback = handler

    @classmethod
    def register_network_handler(cls, handler: Callable[[ApplicationWarning], None]) -> None:
        cls._network_callback = handler

    def __init__(self, name: str):
        super().__init__(name)

    @staticmethod
    def logger_name_for_class(model_class: Type):
        return f"{model_class.__module__}.{model_class.__qualname__}"

    @staticmethod
    def is_testing_mode() -> bool:
        return any(tools in arg for tools in TESTING_TOOLS for arg in sys.argv)

    def notify(self, msg: str):
        from . import INFO

        self.log(INFO, msg)
        if not HummingbotLogger.is_testing_mode() and HummingbotLogger._notify_callback is not None:
            HummingbotLogger._notify_callback(f"({pd.Timestamp.fromtimestamp(int(time.time()))}) {msg}")

    def network(self, log_msg: str, app_warning_msg: str | None = None, *args, **kwargs):
        from . import NETWORK

        self.log(NETWORK, log_msg, *args, **kwargs)
        if app_warning_msg is not None and not HummingbotLogger.is_testing_mode():
            app_warning: ApplicationWarning = ApplicationWarning(
                time.time(), self.name, self.findCaller(), app_warning_msg
            )
            self.warning(app_warning.warning_msg)
            if HummingbotLogger._network_callback is not None:
                HummingbotLogger._network_callback(app_warning)

    #  --- Copied from logging module ---
    def findCaller(self, stack_info=False, stacklevel=1):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        f = currentframe()
        # On some versions of IronPython, currentframe() returns None if
        # IronPython isn't run with -X:Frames.
        if f is not None:
            f = f.f_back
        orig_f = f
        while f and stacklevel > 1:
            f = f.f_back
            stacklevel -= 1
        if not f:
            f = orig_f
        rv: tuple[str, int, str, str | None] = "(unknown file)", 0, "(unknown function)", None
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == _srcfile:
                f = f.f_back
                continue
            sinfo = None
            if stack_info:
                sio = io.StringIO()
                sio.write("Stack (most recent call last):\n")
                traceback.print_stack(f, file=sio)
                sinfo = sio.getvalue()
                if sinfo[-1] == "\n":
                    sinfo = sinfo[:-1]
                sio.close()
            rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
            break
        return rv

    #  --- Copied from logging module ---


_srcfile = os.path.normcase(HummingbotLogger.network.__code__.co_filename)
