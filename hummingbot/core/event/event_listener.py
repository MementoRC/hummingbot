"""Pure-Python port of hummingbot/core/event/event_listener.pyx (C1 conversion).

Dispatch bridge (audit pattern 4):
    PubSub.c_trigger_event (still Cython) calls, at the C level:
        listener.c_set_event_info(event_tag, caller)
        listener.c_call(arg)

    When EventListener was a cdef class those were cdef vtable calls.
    Now that EventListener is a plain Python class, Cython falls back to
    normal Python attribute look-up — which finds these regular ``def``
    methods.  The bridge is preserved: c_call delegates to __call__ exactly
    as the original cdef c_call did.

    Subclasses override __call__; they must NOT define a plain def c_call,
    because the inherited c_call here already routes to __call__.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hummingbot.core.pubsub import PubSub


class EventListener:
    """Base class for all PubSub event listeners.

    Registered with a PubSub instance via add_listener().  When PubSub
    dispatches an event it calls c_set_event_info then c_call on every
    registered listener.  c_call delegates to __call__, which subclasses
    override to implement event handling logic.

    Weak-reference support is provided automatically by Python for plain
    classes (no __slots__ without __weakref__), matching the original
    Cython cdef class that declared ``object __weakref__``.
    """

    def __new__(cls, *args, **kwargs):
        # Mirrors Cython C-level allocation: EventListener instances expose
        # _current_event_tag and _current_event_caller from the moment of
        # allocation, before any subclass __init__ runs. PubSub.c_trigger_event
        # writes both via c_set_event_info() during dispatch.
        instance = super().__new__(cls)
        instance._current_event_tag = 0
        instance._current_event_caller = None
        return instance

    def __init__(self) -> None:
        self._current_event_tag: int = 0
        self._current_event_caller: PubSub | None = None

    # ------------------------------------------------------------------
    # Properties (audit pattern 3 — cdef attribute access)
    # ------------------------------------------------------------------

    @property
    def current_event_tag(self) -> int:
        return self._current_event_tag

    @property
    def current_event_caller(self) -> PubSub | None:
        return self._current_event_caller

    # ------------------------------------------------------------------
    # PubSub dispatch bridge (audit pattern 4)
    # ------------------------------------------------------------------

    def c_set_event_info(self, current_event_tag: int, current_event_caller: PubSub | None) -> None:
        """Called by PubSub before dispatching; injects event context."""
        self._current_event_tag = current_event_tag
        self._current_event_caller = current_event_caller

    def c_call(self, arg: Any) -> None:
        """Called by PubSub to dispatch the event.

        Delegates to __call__ so that subclasses only need to override
        __call__ — the same pattern all existing Python subclasses use.
        """
        self(arg)

    # ------------------------------------------------------------------
    # Abstract dispatch entry point
    # ------------------------------------------------------------------

    def __call__(self, arg: Any) -> None:
        raise NotImplementedError
