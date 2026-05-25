"""B10 — TimeIterator tick event dispatch parity tests.

TimeIterator(PubSub) is the clock-driven base class for strategies, connectors,
and trackers.  It never fires its own named event type; instead its subclasses
call ``c_trigger_event(tag, payload)`` during each tick.

These tests verify that the PubSubBridge delivers identical dispatch semantics
for arbitrary integer event tags — the same tags TimeIterator subclasses use.
Tests drive PubSub and PubSubBridge directly (no Clock/TimeIterator lifecycle)
so they are fast, pure-Python, and free of Cython import requirements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hummingbot.core.event_bus_bridge import PubSubBridge
from hummingbot.core.pubsub import PubSub

from .conftest import RecorderListener, register_on_both

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Synthetic event tags (mirrors the int values subclasses of TimeIterator use)
# ---------------------------------------------------------------------------

TICK_TAG = 100  # generic tick event
ALT_TAG = 200  # second tag for cross-tag isolation tests
PAYLOAD = "tick-payload"


# ---------------------------------------------------------------------------
# Single-listener dispatch parity
# ---------------------------------------------------------------------------


def test_single_listener_receives_payload_on_legacy(pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
    """Legacy PubSub dispatches the payload to a registered listener."""
    legacy, _ = pubsub_pair
    rec = RecorderListener("legacy")
    legacy.add_listener(TICK_TAG, rec)

    legacy.trigger_event(TICK_TAG, PAYLOAD)

    assert rec.calls == [("legacy", PAYLOAD)]


def test_single_listener_receives_payload_on_bridge(pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
    """PubSubBridge dispatches the payload identically to a registered listener."""
    _, bridge = pubsub_pair
    rec = RecorderListener("bridge")
    bridge.add_listener(TICK_TAG, rec)

    bridge.trigger_event(TICK_TAG, PAYLOAD)

    assert rec.calls == [("bridge", PAYLOAD)]


def test_dispatch_parity_via_register_on_both(pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
    """The same listener registered on both sides receives identical calls."""
    legacy, bridge = pubsub_pair
    rec_legacy = RecorderListener("legacy")
    rec_bridge = RecorderListener("bridge")

    register_on_both(legacy, bridge, _FakeEnum(TICK_TAG), rec_legacy)
    register_on_both(legacy, bridge, _FakeEnum(TICK_TAG), rec_bridge)

    legacy.trigger_event(TICK_TAG, PAYLOAD)
    bridge.trigger_event(TICK_TAG, PAYLOAD)

    assert [c[1] for c in rec_legacy.calls] == [PAYLOAD]
    assert [c[1] for c in rec_bridge.calls] == [PAYLOAD]


# ---------------------------------------------------------------------------
# c_trigger_event equivalence (the Cython-level call path)
# ---------------------------------------------------------------------------


def test_c_trigger_event_equivalent_to_trigger_event_on_legacy(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """c_trigger_event and trigger_event produce identical dispatch on PubSub."""
    legacy, _ = pubsub_pair
    rec = RecorderListener("legacy")
    legacy.add_listener(TICK_TAG, rec)

    legacy.c_trigger_event(TICK_TAG, PAYLOAD)

    assert rec.calls == [("legacy", PAYLOAD)]


def test_c_trigger_event_equivalent_to_trigger_event_on_bridge(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """c_trigger_event and trigger_event produce identical dispatch on PubSubBridge."""
    _, bridge = pubsub_pair
    rec = RecorderListener("bridge")
    bridge.add_listener(TICK_TAG, rec)

    bridge.c_trigger_event(TICK_TAG, PAYLOAD)

    assert rec.calls == [("bridge", PAYLOAD)]


# ---------------------------------------------------------------------------
# Cross-tag isolation
# ---------------------------------------------------------------------------


def test_listener_on_tick_tag_does_not_receive_alt_tag_events(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Events on ALT_TAG must not reach a listener subscribed to TICK_TAG — legacy."""
    legacy, _ = pubsub_pair
    rec = RecorderListener("legacy")
    legacy.add_listener(TICK_TAG, rec)

    legacy.trigger_event(ALT_TAG, "other")

    assert rec.calls == []


def test_bridge_listener_on_tick_tag_does_not_receive_alt_tag_events(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Events on ALT_TAG must not reach a listener subscribed to TICK_TAG — bridge."""
    _, bridge = pubsub_pair
    rec = RecorderListener("bridge")
    bridge.add_listener(TICK_TAG, rec)

    bridge.trigger_event(ALT_TAG, "other")

    assert rec.calls == []


# ---------------------------------------------------------------------------
# Multiple sequential ticks
# ---------------------------------------------------------------------------


def test_multiple_ticks_accumulate_in_order_legacy(pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
    """Multiple sequential dispatches accumulate in arrival order — legacy."""
    legacy, _ = pubsub_pair
    rec = RecorderListener("legacy")
    legacy.add_listener(TICK_TAG, rec)

    for i in range(3):
        legacy.trigger_event(TICK_TAG, i)

    assert [c[1] for c in rec.calls] == [0, 1, 2]


def test_multiple_ticks_accumulate_in_order_bridge(pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
    """Multiple sequential dispatches accumulate in arrival order — bridge."""
    _, bridge = pubsub_pair
    rec = RecorderListener("bridge")
    bridge.add_listener(TICK_TAG, rec)

    for i in range(3):
        bridge.trigger_event(TICK_TAG, i)

    assert [c[1] for c in rec.calls] == [0, 1, 2]


# ---------------------------------------------------------------------------
# No-listener dispatch (must not raise)
# ---------------------------------------------------------------------------


def test_trigger_event_with_no_listeners_does_not_raise_legacy(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Dispatching to a tag with no listeners must be a no-op — legacy."""
    legacy, _ = pubsub_pair
    legacy.trigger_event(TICK_TAG, PAYLOAD)  # no listeners registered


def test_trigger_event_with_no_listeners_does_not_raise_bridge(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Dispatching to a tag with no listeners must be a no-op — bridge."""
    _, bridge = pubsub_pair
    bridge.trigger_event(TICK_TAG, PAYLOAD)  # no listeners registered


# ---------------------------------------------------------------------------
# Internal helper (local only — not exported)
# ---------------------------------------------------------------------------


class _FakeEnum:
    """Minimal Enum-shaped object for register_on_both().

    register_on_both() accesses ``.value`` on the enum argument and passes the
    raw int to the bridge.  This lightweight stand-in avoids a real Enum
    definition while keeping the helper's API contract satisfied.
    """

    def __init__(self, value: int) -> None:
        self.value = value
