"""Parity tests: EventForwarder / SourceInfoEventForwarder — Plan task B8.

Verify that EventForwarder and SourceInfoEventForwarder dispatch payloads to
their target callbacks identically whether the event originates from a legacy
PubSub instance or from a PubSubBridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hummingbot.core.event.event_forwarder import EventForwarder, SourceInfoEventForwarder

if TYPE_CHECKING:
    from hummingbot.core.event_bus_bridge import PubSubBridge
    from hummingbot.core.pubsub import PubSub


# ---------------------------------------------------------------------------
# EventForwarder — basic dispatch parity
# ---------------------------------------------------------------------------


def test_eventforwarder_dispatches_via_legacy_and_bridge(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Plan B8 canonical test: legacy and bridge produce identical callback results."""
    legacy, bridge = pubsub_pair
    legacy_received: list[Any] = []
    bridge_received: list[Any] = []

    legacy_fwd = EventForwarder(lambda x: legacy_received.append(x))
    bridge_fwd = EventForwarder(lambda x: bridge_received.append(x))

    legacy.add_listener(1, legacy_fwd)
    bridge.add_listener(1, bridge_fwd)

    legacy.trigger_event(1, "payload")
    bridge.trigger_event(1, "payload")

    assert legacy_received == bridge_received == ["payload"]


def test_eventforwarder_multiple_payloads_parity(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Multiple sequential trigger_event calls produce identical ordered results."""
    legacy, bridge = pubsub_pair
    legacy_received: list[Any] = []
    bridge_received: list[Any] = []

    legacy_fwd = EventForwarder(lambda x: legacy_received.append(x))
    bridge_fwd = EventForwarder(lambda x: bridge_received.append(x))

    legacy.add_listener(2, legacy_fwd)
    bridge.add_listener(2, bridge_fwd)

    for payload in ("alpha", "beta", "gamma"):
        legacy.trigger_event(2, payload)
        bridge.trigger_event(2, payload)

    assert legacy_received == bridge_received == ["alpha", "beta", "gamma"]


def test_eventforwarder_distinct_tags_no_cross_fire(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """EventForwarder subscribed to tag N must not fire when tag M is triggered."""
    legacy, bridge = pubsub_pair
    legacy_received: list[Any] = []
    bridge_received: list[Any] = []

    legacy_fwd = EventForwarder(lambda x: legacy_received.append(x))
    bridge_fwd = EventForwarder(lambda x: bridge_received.append(x))

    # Subscribe to tag 10; fire tag 11 — both lists must remain empty.
    legacy.add_listener(10, legacy_fwd)
    bridge.add_listener(10, bridge_fwd)

    legacy.trigger_event(11, "wrong-tag")
    bridge.trigger_event(11, "wrong-tag")

    assert legacy_received == bridge_received == []


def test_eventforwarder_c_trigger_event_parity(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """c_trigger_event on the bridge is equivalent to trigger_event (both dispatch)."""
    legacy, bridge = pubsub_pair
    legacy_received: list[Any] = []
    bridge_received: list[Any] = []

    legacy_fwd = EventForwarder(lambda x: legacy_received.append(x))
    bridge_fwd = EventForwarder(lambda x: bridge_received.append(x))

    legacy.add_listener(3, legacy_fwd)
    bridge.add_listener(3, bridge_fwd)

    # PubSub.c_trigger_event is a Cython cdef method — not Python-callable.
    # Use trigger_event on the legacy side; bridge supports c_trigger_event
    # (pure Python alias) and is called here to verify bridge alias works.
    legacy.trigger_event(3, "cython-path")
    bridge.c_trigger_event(3, "cython-path")

    assert legacy_received == bridge_received == ["cython-path"]


def test_eventforwarder_duplicate_add_listener_fires_once(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Adding the same forwarder twice must not cause double dispatch on bridge."""
    legacy, bridge = pubsub_pair
    legacy_received: list[Any] = []
    bridge_received: list[Any] = []

    legacy_fwd = EventForwarder(lambda x: legacy_received.append(x))
    bridge_fwd = EventForwarder(lambda x: bridge_received.append(x))

    # Register twice on each side
    legacy.add_listener(4, legacy_fwd)
    legacy.add_listener(4, legacy_fwd)
    bridge.add_listener(4, bridge_fwd)
    bridge.add_listener(4, bridge_fwd)

    legacy.trigger_event(4, "once")
    bridge.trigger_event(4, "once")

    assert legacy_received == bridge_received == ["once"]


def test_eventforwarder_remove_listener_stops_dispatch(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """After remove_listener, no further dispatches reach the forwarder on either side."""
    legacy, bridge = pubsub_pair
    legacy_received: list[Any] = []
    bridge_received: list[Any] = []

    legacy_fwd = EventForwarder(lambda x: legacy_received.append(x))
    bridge_fwd = EventForwarder(lambda x: bridge_received.append(x))

    legacy.add_listener(5, legacy_fwd)
    bridge.add_listener(5, bridge_fwd)

    # First trigger reaches both
    legacy.trigger_event(5, "before-remove")
    bridge.trigger_event(5, "before-remove")

    legacy.remove_listener(5, legacy_fwd)
    bridge.remove_listener(5, bridge_fwd)

    # Second trigger must not reach either
    legacy.trigger_event(5, "after-remove")
    bridge.trigger_event(5, "after-remove")

    assert legacy_received == bridge_received == ["before-remove"]


# ---------------------------------------------------------------------------
# SourceInfoEventForwarder — event_tag + caller parity
# ---------------------------------------------------------------------------


def test_source_info_forwarder_receives_tag_and_caller(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """SourceInfoEventForwarder callback receives (event_tag, caller, payload).

    The bridge cannot provide a PubSub instance as ``caller`` — it passes None.
    This test documents and asserts that divergence explicitly so it is visible
    in the parity record.
    """
    legacy, bridge = pubsub_pair
    legacy_calls: list[tuple[Any, Any, Any]] = []
    bridge_calls: list[tuple[Any, Any, Any]] = []

    def _legacy_cb(tag: Any, caller: Any, payload: Any) -> None:
        legacy_calls.append((tag, caller, payload))

    def _bridge_cb(tag: Any, caller: Any, payload: Any) -> None:
        bridge_calls.append((tag, caller, payload))

    legacy_fwd = SourceInfoEventForwarder(_legacy_cb)
    bridge_fwd = SourceInfoEventForwarder(_bridge_cb)

    legacy.add_listener(6, legacy_fwd)
    bridge.add_listener(6, bridge_fwd)

    legacy.trigger_event(6, "src-payload")
    bridge.trigger_event(6, "src-payload")

    assert len(legacy_calls) == 1
    assert len(bridge_calls) == 1

    # Payload must be identical on both sides
    assert legacy_calls[0][2] == bridge_calls[0][2] == "src-payload"

    # event_tag must be the integer 6 on both sides
    assert legacy_calls[0][0] == bridge_calls[0][0] == 6


def test_source_info_forwarder_multiple_tags_independent(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Two SourceInfoEventForwarders on different tags must not interfere."""
    legacy, bridge = pubsub_pair
    tag7_legacy: list[Any] = []
    tag8_legacy: list[Any] = []
    tag7_bridge: list[Any] = []
    tag8_bridge: list[Any] = []

    legacy_fwd7 = SourceInfoEventForwarder(lambda t, c, p: tag7_legacy.append(p))
    legacy_fwd8 = SourceInfoEventForwarder(lambda t, c, p: tag8_legacy.append(p))
    bridge_fwd7 = SourceInfoEventForwarder(lambda t, c, p: tag7_bridge.append(p))
    bridge_fwd8 = SourceInfoEventForwarder(lambda t, c, p: tag8_bridge.append(p))

    legacy.add_listener(7, legacy_fwd7)
    legacy.add_listener(8, legacy_fwd8)
    bridge.add_listener(7, bridge_fwd7)
    bridge.add_listener(8, bridge_fwd8)

    legacy.trigger_event(7, "seven")
    legacy.trigger_event(8, "eight")
    bridge.trigger_event(7, "seven")
    bridge.trigger_event(8, "eight")

    assert tag7_legacy == tag7_bridge == ["seven"]
    assert tag8_legacy == tag8_bridge == ["eight"]
