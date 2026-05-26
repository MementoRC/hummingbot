"""B6: Parity tests — EventLogger captures events identically via PubSub and PubSubBridge.

Each test drives both the legacy PubSub and the PubSubBridge with an identical
event sequence, then asserts that the two EventLogger instances produce the same
``event_log`` contents and length.

EventLogger.event_log returns ``list(generic_deque) + list(order_fill_deque)``.
All payloads here are plain objects (not OrderFilledEvent), so they land in the
generic deque; order is insertion order.
"""

from __future__ import annotations

from enum import IntEnum
from test.hummingbot.event_bus_parity.conftest import register_on_both
from typing import TYPE_CHECKING

from hummingbot.core.event.event_logger import EventLogger

if TYPE_CHECKING:
    from hummingbot.core.event_bus_bridge import PubSubBridge
    from hummingbot.core.pubsub import PubSub


# ---------------------------------------------------------------------------
# Sentinel tag — avoids collision with real event-tag enums in integration env
# ---------------------------------------------------------------------------


class _Tag(IntEnum):
    ALPHA = 42
    BETA = 99


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loggers() -> tuple[EventLogger, EventLogger]:
    """Return a fresh (legacy_logger, bridge_logger) pair."""
    return EventLogger(), EventLogger()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_eventlogger_records_same_single_event(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Single trigger: both loggers capture exactly one identical event."""
    legacy, bridge = pubsub_pair
    legacy_logger, bridge_logger = _make_loggers()

    legacy.add_listener(_Tag.ALPHA, legacy_logger)
    bridge.add_listener(int(_Tag.ALPHA), bridge_logger)

    payload = {"a": 1}
    legacy.trigger_event(_Tag.ALPHA, payload)
    bridge.trigger_event(int(_Tag.ALPHA), payload)

    assert len(legacy_logger.event_log) == 1
    assert len(bridge_logger.event_log) == 1
    assert legacy_logger.event_log[0] == bridge_logger.event_log[0]


def test_eventlogger_records_same_multiple_events_in_order(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Multiple triggers: event_log order is identical on both sides."""
    legacy, bridge = pubsub_pair
    legacy_logger, bridge_logger = _make_loggers()

    legacy.add_listener(_Tag.ALPHA, legacy_logger)
    bridge.add_listener(int(_Tag.ALPHA), bridge_logger)

    payloads = [{"seq": i} for i in range(5)]
    for p in payloads:
        legacy.trigger_event(_Tag.ALPHA, p)
        bridge.trigger_event(int(_Tag.ALPHA), p)

    assert legacy_logger.event_log == bridge_logger.event_log
    assert legacy_logger.event_log == payloads


def test_eventlogger_independent_per_tag(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Events on tag BETA do not appear in a logger subscribed only to ALPHA."""
    legacy, bridge = pubsub_pair
    legacy_logger, bridge_logger = _make_loggers()

    legacy.add_listener(_Tag.ALPHA, legacy_logger)
    bridge.add_listener(int(_Tag.ALPHA), bridge_logger)

    legacy.trigger_event(_Tag.BETA, {"should": "be ignored"})
    bridge.trigger_event(int(_Tag.BETA), {"should": "be ignored"})

    assert legacy_logger.event_log == []
    assert bridge_logger.event_log == []


def test_eventlogger_via_register_on_both(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """register_on_both helper: a single logger subscribed on both sides receives
    events from both dispatchers, matching the legacy count and payload."""
    legacy, bridge = pubsub_pair
    shared_logger = EventLogger()

    register_on_both(legacy, bridge, _Tag.ALPHA, shared_logger)

    payload = {"x": 10}
    legacy.trigger_event(_Tag.ALPHA, payload)
    bridge.trigger_event(int(_Tag.ALPHA), payload)

    # Shared logger receives from both dispatchers — two entries, same payload
    assert len(shared_logger.event_log) == 2
    assert shared_logger.event_log[0] == payload
    assert shared_logger.event_log[1] == payload


def test_eventlogger_clear_resets_both_sides(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """EventLogger.clear() leaves event_log empty; subsequent events are captured."""
    legacy, bridge = pubsub_pair
    legacy_logger, bridge_logger = _make_loggers()

    legacy.add_listener(_Tag.ALPHA, legacy_logger)
    bridge.add_listener(int(_Tag.ALPHA), bridge_logger)

    legacy.trigger_event(_Tag.ALPHA, {"before": "clear"})
    bridge.trigger_event(int(_Tag.ALPHA), {"before": "clear"})

    legacy_logger.clear()
    bridge_logger.clear()

    assert legacy_logger.event_log == []
    assert bridge_logger.event_log == []

    legacy.trigger_event(_Tag.ALPHA, {"after": "clear"})
    bridge.trigger_event(int(_Tag.ALPHA), {"after": "clear"})

    assert len(legacy_logger.event_log) == 1
    assert len(bridge_logger.event_log) == 1
    assert legacy_logger.event_log[0] == bridge_logger.event_log[0]


def test_eventlogger_remove_listener_stops_capture(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """remove_listener halts capture on both sides consistently."""
    legacy, bridge = pubsub_pair
    legacy_logger, bridge_logger = _make_loggers()

    legacy.add_listener(_Tag.ALPHA, legacy_logger)
    bridge.add_listener(int(_Tag.ALPHA), bridge_logger)

    legacy.trigger_event(_Tag.ALPHA, {"before": "removal"})
    bridge.trigger_event(int(_Tag.ALPHA), {"before": "removal"})

    legacy.remove_listener(_Tag.ALPHA, legacy_logger)
    bridge.remove_listener(int(_Tag.ALPHA), bridge_logger)

    legacy.trigger_event(_Tag.ALPHA, {"after": "removal"})
    bridge.trigger_event(int(_Tag.ALPHA), {"after": "removal"})

    assert len(legacy_logger.event_log) == 1
    assert len(bridge_logger.event_log) == 1
    assert legacy_logger.event_log[0] == bridge_logger.event_log[0]
