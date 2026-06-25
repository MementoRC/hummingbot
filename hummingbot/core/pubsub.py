"""Pure-Python port of hummingbot/core/pubsub.pyx (C12 conversion).

Dispatch bridge (audit pattern 4):
    PubSub.trigger_event is the entry point called by all callers.
    For every registered listener it performs, at the Python level:

        listener.c_set_event_info(event_tag, self)
        listener.c_call(arg)

    Both ``c_set_event_info`` and ``c_call`` are now regular ``def``
    methods on EventListener (see hummingbot/core/event/event_listener.py).
    The contract is identical to the original cdef vtable calls — the
    bridge is preserved end-to-end.

Storage model (audit pattern 1):
    The original .pyx held listeners in a C++ ``unordered_map[int64_t,
    unordered_set[PyRef]]`` (the Events typedef in the deleted .pxd).
    Each PyRef wrapped a CPython weakref via PyWeakref_NewRef.

    The pure-Python port replaces that with::

        self._events: dict[int, list[weakref.ref]]

    indexed by event_tag.  A list (not a set) is used because
    ``weakref.ref`` objects are not stably hashable across listener
    lifetimes in a way that matches the C++ pointer-identity semantics;
    duplicate-listener prevention is preserved by linear identity
    comparison through the referents.  Dead-listener GC semantics are
    preserved verbatim from the original.
"""

from __future__ import annotations

from enum import Enum
import logging
import random
from typing import TYPE_CHECKING, Any
import weakref

from hummingbot.logger import HummingbotLogger

if TYPE_CHECKING:
    from hummingbot.core.event.event_listener import EventListener


class PubSub:
    """PubSub with weak references.

    This avoids the lapsed listener problem by periodically performing GC
    on dead event listeners.

    Dead listener GC is done by calling ``remove_dead_listeners()``,
    which checks whether the listener weak references are alive or not,
    and removes the dead ones.  Each call to ``remove_dead_listeners()``
    takes O(n).

    Here's how the dead listener GC is performed:

    1. ``add_listener()``:
       Randomly with ADD_LISTENER_GC_PROBABILITY.  This assumes
       ``add_listener()`` is called frequently and so it doesn't make
       sense to do the GC every time.
    2. ``remove_listener()``:
       Every time.  This assumes ``remove_listener()`` is called
       infrequently.
    3. ``get_listeners()`` and ``trigger_event()``:
       Every time.  Both functions take O(n) already.
    """

    ADD_LISTENER_GC_PROBABILITY = 0.005

    _logger: HummingbotLogger | None = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __new__(cls, *args, **kwargs):
        # Mirrors Cython's C-level allocation: every PubSub instance has
        # _events as an empty dict from the moment it exists, regardless of
        # whether subclasses call super().__init__(). This preserves the
        # behavior of `cdef Events _events` in the original .pxd, where the
        # C++ unordered_set was default-initialized at allocation time.
        instance = super().__new__(cls)
        instance._events = {}
        return instance

    def __init__(self) -> None:
        self._events: dict[int, list[weakref.ref]] = {}

    # ------------------------------------------------------------------
    # Public Python API
    # ------------------------------------------------------------------

    def add_listener(self, event_tag: Enum, listener: EventListener) -> None:
        tag = event_tag.value
        listeners = self._events.get(tag)
        listener_weakref = weakref.ref(listener)
        if listeners is None:
            self._events[tag] = [listener_weakref]
        else:
            # Preserve original set-like semantics: do not add a duplicate
            # listener (referent identity match).
            for existing in listeners:
                if existing() is listener:
                    break
            else:
                listeners.append(listener_weakref)

        if random.random() < PubSub.ADD_LISTENER_GC_PROBABILITY:
            self.remove_dead_listeners(tag)

    def remove_listener(self, event_tag: Enum, listener: EventListener) -> None:
        tag = event_tag.value
        listeners = self._events.get(tag)
        if listeners is None:
            return
        for idx, existing in enumerate(listeners):
            if existing() is listener:
                del listeners[idx]
                break
        self.remove_dead_listeners(tag)

    def get_listeners(self, event_tag: Enum) -> list[EventListener]:
        tag = event_tag.value
        self.remove_dead_listeners(tag)
        listeners = self._events.get(tag)
        if listeners is None:
            return []
        retval: list[EventListener] = []
        for ref in listeners:
            referent = ref()
            if referent is not None:
                retval.append(referent)
        return retval

    def trigger_event(self, event_tag: Enum, message: Any) -> None:
        tag = event_tag.value
        self.remove_dead_listeners(tag)
        listeners = self._events.get(tag)
        if listeners is None:
            return

        # It is extremely important that this is a snapshot copy of the
        # listener list — listeners are allowed to call
        # ``remove_listener()`` during dispatch, which would otherwise
        # mutate the list we are iterating over.  The original .pyx
        # achieved this with a C++ copy of the unordered_set.
        listeners_snapshot = list(listeners)
        for ref in listeners_snapshot:
            typed_listener = ref()
            if typed_listener is None:
                continue
            try:
                typed_listener.c_set_event_info(tag, self)
                typed_listener.c_call(message)
            except Exception:
                self.c_log_exception(tag, message)
            finally:
                typed_listener.c_set_event_info(0, None)

    def remove_dead_listeners(self, event_tag: int) -> None:
        listeners = self._events.get(event_tag)
        if listeners is None:
            return
        # Match the original .pyx behaviour of randomising removal order
        # (the underlying unordered_set had no defined iteration order).
        random.shuffle(listeners)
        alive = [ref for ref in listeners if ref() is not None]
        if not alive:
            del self._events[event_tag]
        else:
            self._events[event_tag] = alive

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def c_log_exception(self, event_tag: int, arg: Any) -> None:
        self.logger().error(f"Unexpected error while processing event {event_tag}.", exc_info=True)
