"""EventListener base behavior parity tests (Plan task B5).

Verifies:
- ``c_call`` delegates to ``__call__`` (the actual Cython direction: c_call calls self(arg))
- Base ``__call__`` raises NotImplementedError when not overridden
- Subclass overriding ``__call__`` receives dispatch correctly
- RecorderListener (conftest) works identically against PubSub and PubSubBridge
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

import pytest

from hummingbot.core.event.event_listener import EventListener

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Minimal concrete subclasses for isolation
# ---------------------------------------------------------------------------


class _CapturingListener(EventListener):
    """Overrides ``__call__`` to record dispatched values."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[Any] = []

    def __call__(self, arg: Any) -> None:  # type: ignore[override]
        self.received.append(arg)


# ---------------------------------------------------------------------------
# Tests: base-class __call__ behavior
# ---------------------------------------------------------------------------


def test_base_event_listener_call_raises_not_implemented() -> None:
    """Unsubclassed EventListener.__call__ must raise NotImplementedError."""
    listener = EventListener()
    with pytest.raises(NotImplementedError):
        listener("any-arg")


def test_subclass_call_receives_argument() -> None:
    """Subclass overriding __call__ receives the dispatched argument."""
    listener = _CapturingListener()
    listener("hello")
    assert listener.received == ["hello"]


def test_subclass_call_receives_multiple_arguments_in_order() -> None:
    """Multiple dispatches are recorded in dispatch order."""
    listener = _CapturingListener()
    for value in (1, "two", None, {"k": "v"}):
        listener(value)
    assert listener.received == [1, "two", None, {"k": "v"}]


# ---------------------------------------------------------------------------
# Tests: c_call delegates to __call__
# Note: c_call is a cdef method (C-level) and is not directly callable from
# Python; it is invoked internally by PubSub when it dispatches events.
# We verify the delegation indirectly via PubSub.trigger_event.
# ---------------------------------------------------------------------------


class _SentinelTag(IntEnum):
    TICK = 42


def test_c_call_delegates_to_call_via_pubsub(pubsub_pair: tuple) -> None:
    """c_call (invoked by PubSub) ultimately calls __call__ on the listener.

    PubSub.trigger_event → c_call (Cython) → self(arg) → __call__.
    The capturing subclass proves the full chain completes.
    """
    legacy, _bridge = pubsub_pair
    listener = _CapturingListener()
    legacy.add_listener(_SentinelTag.TICK, listener)
    legacy.trigger_event(_SentinelTag.TICK, "via-c-call")
    assert listener.received == ["via-c-call"]


def test_c_call_delegation_via_bridge(pubsub_pair: tuple) -> None:
    """Same delegation chain holds when the bridge dispatches the event."""
    _legacy, bridge = pubsub_pair
    listener = _CapturingListener()
    bridge.add_listener(_SentinelTag.TICK.value, listener)
    bridge.trigger_event(_SentinelTag.TICK.value, "via-bridge")
    assert listener.received == ["via-bridge"]


# ---------------------------------------------------------------------------
# Tests: parity between legacy PubSub and PubSubBridge
# ---------------------------------------------------------------------------


def test_listener_receives_same_payload_from_legacy_and_bridge(
    pubsub_pair: tuple,
) -> None:
    """Legacy PubSub and PubSubBridge deliver identical payloads to listeners."""
    legacy, bridge = pubsub_pair
    legacy_listener = _CapturingListener()
    bridge_listener = _CapturingListener()

    legacy.add_listener(_SentinelTag.TICK, legacy_listener)
    bridge.add_listener(_SentinelTag.TICK.value, bridge_listener)

    payload = {"price": 100.0, "qty": 5}
    legacy.trigger_event(_SentinelTag.TICK, payload)
    bridge.trigger_event(_SentinelTag.TICK.value, payload)

    assert legacy_listener.received == bridge_listener.received == [payload]


def test_listener_call_count_parity(pubsub_pair: tuple) -> None:
    """Listener is invoked exactly once per trigger on both sides."""
    legacy, bridge = pubsub_pair
    legacy_listener = _CapturingListener()
    bridge_listener = _CapturingListener()

    legacy.add_listener(_SentinelTag.TICK, legacy_listener)
    bridge.add_listener(_SentinelTag.TICK.value, bridge_listener)

    for _ in range(3):
        legacy.trigger_event(_SentinelTag.TICK, "tick")
        bridge.trigger_event(_SentinelTag.TICK.value, "tick")

    assert len(legacy_listener.received) == len(bridge_listener.received) == 3


def test_listener_not_called_after_removal_legacy(pubsub_pair: tuple) -> None:
    """Listener removed from legacy PubSub stops receiving events."""
    legacy, _bridge = pubsub_pair
    listener = _CapturingListener()
    legacy.add_listener(_SentinelTag.TICK, listener)
    legacy.trigger_event(_SentinelTag.TICK, "before")
    legacy.remove_listener(_SentinelTag.TICK, listener)
    legacy.trigger_event(_SentinelTag.TICK, "after")
    assert listener.received == ["before"]


def test_listener_not_called_after_removal_bridge(pubsub_pair: tuple) -> None:
    """Listener removed from bridge stops receiving events."""
    _legacy, bridge = pubsub_pair
    listener = _CapturingListener()
    bridge.add_listener(_SentinelTag.TICK.value, listener)
    bridge.trigger_event(_SentinelTag.TICK.value, "before")
    bridge.remove_listener(_SentinelTag.TICK.value, listener)
    bridge.trigger_event(_SentinelTag.TICK.value, "after")
    assert listener.received == ["before"]


# ---------------------------------------------------------------------------
# Tests: current_event_tag / current_event_caller properties
# ---------------------------------------------------------------------------


def test_event_listener_properties_initial_state() -> None:
    """Fresh EventListener has zero tag and None caller."""
    listener = EventListener()
    assert listener.current_event_tag == 0
    assert listener.current_event_caller is None
