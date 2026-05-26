#!/usr/bin/env python

from datetime import datetime
from logging import StreamHandler
from typing import TextIO


class CLIHandler(StreamHandler[TextIO]):
    def formatException(self, _) -> str | None:
        return None

    def format(self, record) -> str:
        exc_info = record.exc_info
        if record.exc_info is not None:
            record.exc_info = None
        retval = (
            f"{datetime.fromtimestamp(record.created).strftime('%H:%M:%S')} - {record.name.split('.')[-1]} - "
            f"{record.getMessage()}"
        )
        if exc_info:
            retval += " (See log file for stack trace dump)"
        record.exc_info = exc_info
        return retval
