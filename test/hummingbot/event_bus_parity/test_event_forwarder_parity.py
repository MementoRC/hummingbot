"""Parity tests: EventForwarder / SourceInfoEventForwarder — Plan task B8.

Verify that EventForwarder and SourceInfoEventForwarder dispatch payloads to
their target callbacks identically whether the event originates from a legacy
PubSub instance or from a PubSubBridge.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

from hummingbot.core.event.event_forwarder import EventForwarder, SourceInfoEventForwarder

if TYPE_CHECKING:
    from hummingbot.core.event_bus_bridge import PubSubBridge
    from hummingbot.core.pubsub import PubSub


class _FwdTag(IntEnum):
    """Sentinel tags for EventForwarder parity tests.

    PubSub.trigger_event expects an Enum (calls .value internally).
    PubSubBridge.trigger_event expects an int.
    Using IntEnum members satisfies legacy; .value satisfies bridge.
    """

    T1 = 1
    T2 = 2
    T3 = 3
    T4 = 4
    T5 = 5
    T6 = 6
    T7 = 7
    T8 = 8
    T10 = 10
    T11 = 11


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

    legacy.add_listener(_FwdTag.T1, legacy_fwd)
    bridge.add_listener(_FwdTag.T1.value, bridge_fwd)

    legacy.trigger_event(_FwdTag.T1, "payload")
    bridge.trigger_event(_FwdTag.T1.value, "payload")

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

    legacy.add_listener(_FwdTag.T2, legacy_fwd)
    bridge.add_listener(_FwdTag.T2.value, bridge_fwd)

    for payload in ("alpha", "beta", "gamma"):
        legacy.trigger_event(_FwdTag.T2, payload)
        bridge.trigger_event(_FwdTag.T2.value, payload)

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
    legacy.add_listener(_FwdTag.T10, legacy_fwd)
    bridge.add_listener(_FwdTag.T10.value, bridge_fwd)

    legacy.trigger_event(_FwdTag.T11, "wrong-tag")
    bridge.trigger_event(_FwdTag.T11.value, "wrong-tag")

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

    legacy.add_listener(_FwdTag.T3, legacy_fwd)
    bridge.add_listener(_FwdTag.T3.value, bridge_fwd)

    # PubSub.c_trigger_event is a Cython cdef method — not Python-callable.
    # Use trigger_event on the legacy side; bridge supports c_trigger_event
    # (pure Python alias) and is called here to verify bridge alias works.
    legacy.trigger_event(_FwdTag.T3, "cython-path")
    bridge.c_trigger_event(_FwdTag.T3.value, "cython-path")

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
    legacy.add_listener(_FwdTag.T4, legacy_fwd)
    legacy.add_listener(_FwdTag.T4, legacy_fwd)
    bridge.add_listener(_FwdTag.T4.value, bridge_fwd)
    bridge.add_listener(_FwdTag.T4.value, bridge_fwd)

    legacy.trigger_event(_FwdTag.T4, "once")
    bridge.trigger_event(_FwdTag.T4.value, "once")

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

    legacy.add_listener(_FwdTag.T5, legacy_fwd)
    bridge.add_listener(_FwdTag.T5.value, bridge_fwd)

    # First trigger reaches both
    legacy.trigger_event(_FwdTag.T5, "before-remove")
    bridge.trigger_event(_FwdTag.T5.value, "before-remove")

    legacy.remove_listener(_FwdTag.T5, legacy_fwd)
    bridge.remove_listener(_FwdTag.T5.value, bridge_fwd)

    # Second trigger must not reach either
    legacy.trigger_event(_FwdTag.T5, "after-remove")
    bridge.trigger_event(_FwdTag.T5.value, "after-remove")

    assert legacy_received == bridge_received == ["before-remove"]


# ---------------------------------------------------------------------------
# SourceInfoEventForwarder — event_tag + caller parity
# ---------------------------------------------------------------------------


def test_source_info_forwarder_receives_tag_and_caller(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """SourceInfoEventForwarder callback receives (event_tag, caller, payload).

    Phase B documented divergence: PubSub sets ``current_event_tag`` and
    ``current_event_caller`` on each EventListener before dispatching via the
    Cython ``c_set_event_info`` cdef method.  PubSubBridge invokes the listener
    directly via ``listener(payload)`` without setting those fields, so:

    - ``current_event_tag`` is 0 (EventListener default) on the bridge side.
    - ``current_event_caller`` is None on both sides (legacy sets it to the
      PubSub instance, but SourceInfoEventForwarder exposes it opaquely).

    Payload delivery itself is identical.  The tag divergence is an inherent
    Phase B limitation and is tracked for Phase C resolution (direct EventBus
    API will carry the tag natively).
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

    legacy.add_listener(_FwdTag.T6, legacy_fwd)
    bridge.add_listener(_FwdTag.T6.value, bridge_fwd)

    legacy.trigger_event(_FwdTag.T6, "src-payload")
    bridge.trigger_event(_FwdTag.T6.value, "src-payload")

    assert len(legacy_calls) == 1
    assert len(bridge_calls) == 1

    # Payload must be identical on both sides (core parity property).
    assert legacy_calls[0][2] == bridge_calls[0][2] == "src-payload"

    # Phase B divergence: legacy sets current_event_tag via c_set_event_info;
    # bridge does not — the field retains its EventListener default of 0.
    assert legacy_calls[0][0] == 6, f"legacy tag: expected 6, got {legacy_calls[0][0]!r}"
    assert bridge_calls[0][0] == 0, (
        f"bridge tag: expected 0 (Phase B limitation — bridge does not set "
        f"current_event_tag), got {bridge_calls[0][0]!r}"
    )


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

    legacy.add_listener(_FwdTag.T7, legacy_fwd7)
    legacy.add_listener(_FwdTag.T8, legacy_fwd8)
    bridge.add_listener(_FwdTag.T7.value, bridge_fwd7)
    bridge.add_listener(_FwdTag.T8.value, bridge_fwd8)

    legacy.trigger_event(_FwdTag.T7, "seven")
    legacy.trigger_event(_FwdTag.T8, "eight")
    bridge.trigger_event(_FwdTag.T7.value, "seven")
    bridge.trigger_event(_FwdTag.T8.value, "eight")

    assert tag7_legacy == tag7_bridge == ["seven"]
    assert tag8_legacy == tag8_bridge == ["eight"]
