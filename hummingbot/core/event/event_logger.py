"""Pure-Python port of hummingbot/core/event/event_logger.pyx (C2 conversion).

Dispatch convention (audit pattern 4):
    EventListener.c_call(arg) at the C level ultimately calls self(arg), i.e.
    __call__.  Python subclasses cannot override a cdef method at the C level,
    so the entire dispatch logic lives in __call__ — no c_call delegation.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Optional

from async_timeout import timeout as async_timeout

from hummingbot.core.event.event_listener import EventListener
from hummingbot.core.event.events import OrderFilledEvent


class EventLogger(EventListener):
    """Captures PubSub events for inspection in tests and debugging.

    EventLogger is registered as an EventListener on a PubSub instance via
    add_listener().  When PubSub dispatches an event it calls c_call on the
    listener, which at the C level routes to self(arg) == __call__.

    Order-fill events are kept in an unbounded deque (required for PnL
    calculation); all other events are kept in a bounded deque (maxlen=50).
    """

    def __init__(self, event_source: Optional[str] = None) -> None:
        super().__init__()
        self._event_source: Optional[str] = event_source
        # Bounded — keeps only the 50 most recent non-fill events
        self._generic_logged_events: deque[Any] = deque(maxlen=50)
        # Unbounded — every fill event must be retained for PnL accounting
        self._order_filled_logged_events: deque[Any] = deque()
        self._logged_events: dict[type, deque[Any]] = {
            OrderFilledEvent: self._order_filled_logged_events,
        }
        self._waiting: dict[asyncio.Event, type] = {}
        self._wait_returns: dict[asyncio.Event, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def event_log(self) -> list[Any]:
        """All captured events: generic (oldest first) then order-fills."""
        return list(self._generic_logged_events) + list(self._order_filled_logged_events)

    @property
    def event_source(self) -> Optional[str]:
        return self._event_source

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Discard all captured events; leaves listeners intact."""
        self._generic_logged_events.clear()
        self._order_filled_logged_events.clear()

    # ------------------------------------------------------------------
    # Async helpers
    # ------------------------------------------------------------------

    async def wait_for(self, event_type: type, timeout_seconds: float = 180) -> Any:
        """Block until an event of ``event_type`` is dispatched; return it.

        Raises:
            asyncio.TimeoutError: if no matching event arrives within
                ``timeout_seconds``.
        """
        notifier = asyncio.Event()
        self._waiting[notifier] = event_type

        async with async_timeout(timeout_seconds):
            await notifier.wait()

        retval = self._wait_returns.get(notifier)
        if notifier in self._wait_returns:
            del self._wait_returns[notifier]
        del self._waiting[notifier]
        return retval

    # ------------------------------------------------------------------
    # Dispatch entry point (audit pattern 4)
    # ------------------------------------------------------------------

    def __call__(self, event_object: Any) -> None:
        """Receive a dispatched event.

        This is the canonical entry point for both direct Python calls and the
        Cython PubSub dispatch path (EventListener.c_call → self(arg)).  A
        plain ``def c_call`` on a Python subclass would NOT override the
        inherited ``cdef c_call`` at the C level, so all logic lives here.
        """
        self._logged_events.get(type(event_object), self._generic_logged_events).append(event_object)
        event_object_type = type(event_object)

        should_notify = []
        for notifier, waiting_event_type in self._waiting.items():
            if event_object_type is waiting_event_type:
                should_notify.append(notifier)
                self._wait_returns[notifier] = event_object
        for notifier in should_notify:
            notifier.set()
