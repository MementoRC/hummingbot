#!/usr/bin/env python

from typing import NamedTuple


class ApplicationWarning(NamedTuple):
    timestamp: float
    logger_name: str
    caller_info: tuple[str, int, str, str | None]
    warning_msg: str

    @property
    def filename(self) -> str:
        return self.caller_info[0]

    @property
    def line_number(self) -> int:
        return self.caller_info[1]

    @property
    def function_name(self) -> str:
        return self.caller_info[2]

    @property
    def stack_info(self) -> str | None:
        return self.caller_info[3]
