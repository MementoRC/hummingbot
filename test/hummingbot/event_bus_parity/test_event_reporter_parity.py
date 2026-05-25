"""Parity tests: EventReporter under legacy PubSub vs PubSubBridge.

Plan task B7 — verifies that EventReporter dispatches the same structured
log payload to its logger regardless of whether the underlying event bus is
the legacy PubSub or the PubSubBridge shim.

EventReporter does not accumulate state; it calls ``self.logger().event_log``
once per dispatched event.  Parity is asserted by capturing those calls with a
unittest.mock and comparing the positional argument (the event dict) across the
two sides.
"""

from __future__ import annotations

import dataclasses
from collections import namedtuple
from enum import IntEnum
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from hummingbot.core.event.event_reporter import EventReporter

if TYPE_CHECKING:
    from hummingbot.core.event_bus_bridge import PubSubBridge
    from hummingbot.core.pubsub import PubSub

# ---------------------------------------------------------------------------
# Minimal event fixtures
# ---------------------------------------------------------------------------


class _ReporterTag(IntEnum):
    """Sentinel tag for reporter parity tests.

    PubSub.trigger_event expects an Enum (calls .value internally).
    PubSubBridge.trigger_event expects an int.
    Using an IntEnum satisfies both: pass _ReporterTag.TEST to legacy,
    pass int(_ReporterTag.TEST) to bridge.
    """

    TEST = 99


_TAG = _ReporterTag.TEST


@dataclasses.dataclass
class _DataclassEvent:
    value: int
    label: str = "dc"


# Named-tuple style event (uses ._asdict() branch in EventReporter.c_call)
_NTEvent = namedtuple("_NTEvent", ["value", "label"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reporter_dataclass_event_same_dict_legacy_and_bridge(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Dataclass event produces identical log dict via legacy and bridge."""
    legacy, bridge = pubsub_pair
    legacy_reporter = EventReporter(event_source="legacy")
    bridge_reporter = EventReporter(event_source="legacy")

    mock_log_legacy = MagicMock()
    mock_log_bridge = MagicMock()

    event = _DataclassEvent(value=7)

    with (
        patch.object(legacy_reporter, "logger", return_value=mock_log_legacy),
        patch.object(bridge_reporter, "logger", return_value=mock_log_bridge),
    ):
        legacy.add_listener(_TAG, legacy_reporter)
        bridge.add_listener(int(_TAG), bridge_reporter)
        legacy.trigger_event(_TAG, event)
        bridge.trigger_event(int(_TAG), event)

    assert mock_log_legacy.event_log.call_count == 1
    assert mock_log_bridge.event_log.call_count == 1

    legacy_dict = mock_log_legacy.event_log.call_args.args[0]
    bridge_dict = mock_log_bridge.event_log.call_args.args[0]

    assert legacy_dict == bridge_dict, f"Log payload mismatch:\nlegacy: {legacy_dict!r}\nbridge: {bridge_dict!r}"


def test_reporter_namedtuple_event_same_dict_legacy_and_bridge(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Named-tuple event (._asdict() branch) produces identical log dict via legacy and bridge."""
    legacy, bridge = pubsub_pair
    legacy_reporter = EventReporter(event_source="src-nt")
    bridge_reporter = EventReporter(event_source="src-nt")

    mock_log_legacy = MagicMock()
    mock_log_bridge = MagicMock()

    event = _NTEvent(value=3, label="nt")

    with (
        patch.object(legacy_reporter, "logger", return_value=mock_log_legacy),
        patch.object(bridge_reporter, "logger", return_value=mock_log_bridge),
    ):
        legacy.add_listener(_TAG, legacy_reporter)
        bridge.add_listener(int(_TAG), bridge_reporter)
        legacy.trigger_event(_TAG, event)
        bridge.trigger_event(int(_TAG), event)

    assert mock_log_legacy.event_log.call_count == 1
    assert mock_log_bridge.event_log.call_count == 1

    legacy_dict = mock_log_legacy.event_log.call_args.args[0]
    bridge_dict = mock_log_bridge.event_log.call_args.args[0]

    assert legacy_dict == bridge_dict, f"Log payload mismatch:\nlegacy: {legacy_dict!r}\nbridge: {bridge_dict!r}"


def test_reporter_dict_includes_event_name_and_source(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Both sides inject event_name and event_source keys into the log dict."""
    legacy, bridge = pubsub_pair
    legacy_reporter = EventReporter(event_source="unit-test")
    bridge_reporter = EventReporter(event_source="unit-test")

    mock_log_legacy = MagicMock()
    mock_log_bridge = MagicMock()

    event = _DataclassEvent(value=42)

    with (
        patch.object(legacy_reporter, "logger", return_value=mock_log_legacy),
        patch.object(bridge_reporter, "logger", return_value=mock_log_bridge),
    ):
        legacy.add_listener(_TAG, legacy_reporter)
        bridge.add_listener(int(_TAG), bridge_reporter)
        legacy.trigger_event(_TAG, event)
        bridge.trigger_event(int(_TAG), event)

    for side, mock_log in [("legacy", mock_log_legacy), ("bridge", mock_log_bridge)]:
        logged = mock_log.event_log.call_args.args[0]
        assert "event_name" in logged, f"{side}: missing 'event_name' key"
        assert logged["event_name"] == "_DataclassEvent", f"{side}: wrong event_name"
        assert "event_source" in logged, f"{side}: missing 'event_source' key"
        assert logged["event_source"] == "unit-test", f"{side}: wrong event_source"


def test_reporter_multiple_events_count_matches(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Dispatching N events fires logger.event_log N times on each side."""
    legacy, bridge = pubsub_pair
    legacy_reporter = EventReporter(event_source="multi")
    bridge_reporter = EventReporter(event_source="multi")

    mock_log_legacy = MagicMock()
    mock_log_bridge = MagicMock()

    n = 5
    events = [_DataclassEvent(value=i) for i in range(n)]

    with (
        patch.object(legacy_reporter, "logger", return_value=mock_log_legacy),
        patch.object(bridge_reporter, "logger", return_value=mock_log_bridge),
    ):
        legacy.add_listener(_TAG, legacy_reporter)
        bridge.add_listener(int(_TAG), bridge_reporter)
        for ev in events:
            legacy.trigger_event(_TAG, ev)
            bridge.trigger_event(int(_TAG), ev)

    assert mock_log_legacy.event_log.call_count == n
    assert mock_log_bridge.event_log.call_count == n

    # All payloads must be pairwise equal.
    legacy_dicts = [c.args[0] for c in mock_log_legacy.event_log.call_args_list]
    bridge_dicts = [c.args[0] for c in mock_log_bridge.event_log.call_args_list]
    assert legacy_dicts == bridge_dicts


def test_reporter_no_call_after_remove_listener(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Removing reporter from both sides silences logging on both."""
    legacy, bridge = pubsub_pair
    legacy_reporter = EventReporter(event_source="removal")
    bridge_reporter = EventReporter(event_source="removal")

    mock_log_legacy = MagicMock()
    mock_log_bridge = MagicMock()

    event = _DataclassEvent(value=1)

    with (
        patch.object(legacy_reporter, "logger", return_value=mock_log_legacy),
        patch.object(bridge_reporter, "logger", return_value=mock_log_bridge),
    ):
        legacy.add_listener(_TAG, legacy_reporter)
        bridge.add_listener(int(_TAG), bridge_reporter)

        # Fire once — should log.
        legacy.trigger_event(_TAG, event)
        bridge.trigger_event(int(_TAG), event)

        assert mock_log_legacy.event_log.call_count == 1
        assert mock_log_bridge.event_log.call_count == 1

        # Remove then fire again — should NOT log.
        legacy.remove_listener(_TAG, legacy_reporter)
        bridge.remove_listener(int(_TAG), bridge_reporter)
        legacy.trigger_event(_TAG, event)
        bridge.trigger_event(int(_TAG), event)

    assert mock_log_legacy.event_log.call_count == 1, "legacy called after removal"
    assert mock_log_bridge.event_log.call_count == 1, "bridge called after removal"
