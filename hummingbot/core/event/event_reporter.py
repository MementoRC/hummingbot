"""Pure-Python port of hummingbot/core/event/event_reporter.pyx (C3 conversion).

Dispatch convention (audit pattern 4):
    EventListener.c_call(arg) at the C level ultimately calls self(arg), i.e.
    __call__.  Python subclasses cannot override a cdef method at the C level,
    so the entire dispatch logic lives in __call__ — no c_call delegation.

Dispatch variant: B — c_call body absorbed directly into __call__ (same as C2
EventLogger; EventReporter has no separate helper methods).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Optional

from hummingbot.core.event.event_listener import EventListener

er_logger: Optional[logging.Logger] = None


class EventReporter(EventListener):
    """Event listener that logs events to a structured logger.

    Registered on a PubSub instance via add_listener().  Each dispatched event
    is serialised to a dict and forwarded to ``self.logger().event_log``.
    """

    def __init__(self, event_source: Optional[str] = None) -> None:
        super().__init__()
        self.event_source: Optional[str] = event_source

    @classmethod
    def logger(cls) -> logging.Logger:
        global er_logger
        if er_logger is None:
            er_logger = logging.getLogger(__name__)
        return er_logger

    # ------------------------------------------------------------------
    # Dispatch entry point (audit pattern 4)
    # ------------------------------------------------------------------

    def __call__(self, event_object: Any) -> None:
        """Receive a dispatched event and log it as a structured dict.

        This is the canonical entry point for both direct Python calls and the
        Cython PubSub dispatch path (EventListener.c_call → self(arg)).  A
        plain ``def c_call`` on a Python subclass would NOT override the
        inherited ``cdef c_call`` at the C level, so all logic lives here.
        """
        try:
            if dataclasses.is_dataclass(event_object):
                event_dict = dataclasses.asdict(event_object)
            else:
                event_dict = event_object._asdict()

            event_dict.update(
                {
                    "event_name": event_object.__class__.__name__,
                    "event_source": self.event_source,
                }
            )
            self.logger().event_log(event_dict)  # type: ignore[attr-defined]
        except Exception:
            self.logger().error("Error logging events.", exc_info=True)
