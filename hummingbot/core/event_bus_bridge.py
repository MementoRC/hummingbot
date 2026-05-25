"""Transient bridge wrapping the canonical EventBus with the legacy PubSub-shaped API.

Lives during Phase B only. Deleted in Phase C once Cython consumers are refactored
to use event_bus.EventBus directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary

from event_bus import EventBus, Subscription

if TYPE_CHECKING:
    from hummingbot.core.event.event_listener import EventListener


class PubSubBridge:
    """Wraps EventBus, exposing the legacy PubSub API (add_listener / remove_listener /
    get_listeners / trigger_event / c_trigger_event).

    Int event tags are translated to string topics via ``{name_prefix}:{event_tag}``
    before being passed to the underlying EventBus.  This is the only layer of
    translation; Phase C removes it entirely.
    """

    def __init__(
        self,
        bus: EventBus | None = None,
        *,
        name_prefix: str = "market",
    ) -> None:
        self._bus: EventBus = bus if bus is not None else EventBus(name="pubsub-bridge")
        self._name_prefix: str = name_prefix
        # (event_tag, listener_id) -> Subscription on the underlying bus
        self._subs: dict[tuple[int, int], Subscription] = {}
        # WeakValueDictionary mirrors legacy PubSub's weak-ref semantics for listeners
        self._listeners: WeakValueDictionary[int, EventListener] = WeakValueDictionary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _topic(self, event_tag: int) -> str:
        """Translate an int event tag to an EventBus string topic."""
        return f"{self._name_prefix}:{event_tag}"

    def _make_callback(self, listener: EventListener) -> Any:
        """Return a closure that routes EventBus payloads to the legacy listener."""

        def cb(payload: Any) -> None:
            if hasattr(listener, "c_call"):
                # Cython EventListener subclass — c_call is the actual dispatch method.
                # EventListener.__call__ raises NotImplementedError; only c_call is
                # overridden by subclasses (EventLogger, EventReporter, EventForwarder).
                listener.c_call(payload)
            else:
                # Plain Python callable.
                listener(payload)

        return cb

    # ------------------------------------------------------------------
    # Legacy PubSub public API
    # ------------------------------------------------------------------

    def add_listener(self, event_tag: int, listener: EventListener) -> None:
        """Subscribe *listener* to *event_tag*.  Duplicate registrations are no-ops
        (matches legacy PubSub semantics where the same listener appears once per tag).
        """
        listener_id = id(listener)
        if (event_tag, listener_id) in self._subs:
            return
        self._listeners[listener_id] = listener
        sub = self._bus.subscribe(self._topic(event_tag), self._make_callback(listener))
        self._subs[(event_tag, listener_id)] = sub

    def remove_listener(self, event_tag: int, listener: EventListener) -> None:
        """Unsubscribe *listener* from *event_tag*.  Idempotent."""
        sub = self._subs.pop((event_tag, id(listener)), None)
        if sub is not None:
            sub.cancel()

    def get_listeners(self, event_tag: int) -> list[EventListener]:
        """Return live listeners registered for *event_tag*."""
        return [self._listeners[lid] for tag, lid in self._subs if tag == event_tag and lid in self._listeners]

    def trigger_event(self, event_tag: int, message: Any) -> None:
        """Dispatch *message* to all listeners registered for *event_tag*."""
        self._bus.publish(self._topic(event_tag), message)

    def c_trigger_event(self, event_tag: int, message: Any) -> None:
        """Cython-callable variant — delegates to trigger_event.

        In the legacy Cython class, c_trigger_event is the C-level implementation
        and trigger_event calls it.  In this pure-Python bridge the relationship is
        reversed: both are equivalent entry points.
        """
        self.trigger_event(event_tag, message)
