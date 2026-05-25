"""Parity tests: EventReporter under legacy PubSub vs PubSubBridge.

Plan task B7 — verifies that EventReporter dispatches the same structured
log payload to its logger regardless of whether the underlying event bus is
the legacy PubSub or the PubSubBridge shim.

EventReporter does not accumulate state; it calls ``self.logger().event_log``
once per dispatched event.  Parity is asserted by capturing those calls with a
unittest.mock and comparing the positional argument (the event dict) across the
two sides.

Implementation note — Cython classmethod patching
--------------------------------------------------
EventReporter.logger() is a @classmethod that returns a module-level singleton
``er_logger``.  EventReporter.c_call() (compiled Cython) calls ``self.logger()``
through the C-level vtable, bypassing ``patch.object(instance, "logger", ...)``.
The correct interception point is the module-level ``er_logger`` variable in
``hummingbot.core.event.event_reporter``.  All tests here patch that variable
directly so that c_call() picks up the mock via the classmethod return value.
"""

from __future__ import annotations

import dataclasses
from collections import namedtuple
from enum import IntEnum
from typing import TYPE_CHECKING
from unittest.mock import patch

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

# Module path for the er_logger singleton (Cython classmethod return value).
_ER_LOGGER_PATH = "hummingbot.core.event.event_reporter.er_logger"


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

    event = _DataclassEvent(value=7)

    # Patch the module-level er_logger singleton; c_call reaches it via classmethod.
    with patch(_ER_LOGGER_PATH) as mock_log:
        legacy.add_listener(_TAG, legacy_reporter)
        bridge.add_listener(int(_TAG), bridge_reporter)
        legacy.trigger_event(_TAG, event)
        bridge.trigger_event(int(_TAG), event)

    # Both reporters called the same logger; total call count must be 2.
    assert mock_log.event_log.call_count == 2, (
        f"Expected 2 event_log calls (one per side), got {mock_log.event_log.call_count}"
    )
    legacy_dict = mock_log.event_log.call_args_list[0].args[0]
    bridge_dict = mock_log.event_log.call_args_list[1].args[0]

    assert legacy_dict == bridge_dict, f"Log payload mismatch:\nlegacy: {legacy_dict!r}\nbridge: {bridge_dict!r}"


def test_reporter_namedtuple_event_same_dict_legacy_and_bridge(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Named-tuple event (._asdict() branch) produces identical log dict via legacy and bridge."""
    legacy, bridge = pubsub_pair
    legacy_reporter = EventReporter(event_source="src-nt")
    bridge_reporter = EventReporter(event_source="src-nt")

    event = _NTEvent(value=3, label="nt")

    with patch(_ER_LOGGER_PATH) as mock_log:
        legacy.add_listener(_TAG, legacy_reporter)
        bridge.add_listener(int(_TAG), bridge_reporter)
        legacy.trigger_event(_TAG, event)
        bridge.trigger_event(int(_TAG), event)

    assert mock_log.event_log.call_count == 2, (
        f"Expected 2 event_log calls (one per side), got {mock_log.event_log.call_count}"
    )
    legacy_dict = mock_log.event_log.call_args_list[0].args[0]
    bridge_dict = mock_log.event_log.call_args_list[1].args[0]

    assert legacy_dict == bridge_dict, f"Log payload mismatch:\nlegacy: {legacy_dict!r}\nbridge: {bridge_dict!r}"


def test_reporter_dict_includes_event_name_and_source(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Both sides inject event_name and event_source keys into the log dict."""
    legacy, bridge = pubsub_pair
    legacy_reporter = EventReporter(event_source="unit-test")
    bridge_reporter = EventReporter(event_source="unit-test")

    event = _DataclassEvent(value=42)

    with patch(_ER_LOGGER_PATH) as mock_log:
        legacy.add_listener(_TAG, legacy_reporter)
        bridge.add_listener(int(_TAG), bridge_reporter)
        legacy.trigger_event(_TAG, event)
        bridge.trigger_event(int(_TAG), event)

    assert mock_log.event_log.call_count == 2
    for i, side in enumerate(("legacy", "bridge")):
        logged = mock_log.event_log.call_args_list[i].args[0]
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

    n = 5
    events = [_DataclassEvent(value=i) for i in range(n)]

    with patch(_ER_LOGGER_PATH) as mock_log:
        legacy.add_listener(_TAG, legacy_reporter)
        bridge.add_listener(int(_TAG), bridge_reporter)
        for ev in events:
            legacy.trigger_event(_TAG, ev)
            bridge.trigger_event(int(_TAG), ev)

    # n events × 2 reporters sharing the same logger module singleton = 2n calls.
    assert mock_log.event_log.call_count == 2 * n, f"Expected {2 * n} calls, got {mock_log.event_log.call_count}"

    # Paired calls must match: legacy call i vs bridge call i.
    all_dicts = [c.args[0] for c in mock_log.event_log.call_args_list]
    legacy_dicts = all_dicts[::2]  # calls 0,2,4,...  (legacy fires first each iteration)
    bridge_dicts = all_dicts[1::2]  # calls 1,3,5,...
    assert legacy_dicts == bridge_dicts, "Pairwise log dicts must match across legacy and bridge"


def test_reporter_no_call_after_remove_listener(
    pubsub_pair: tuple[PubSub, PubSubBridge],
) -> None:
    """Removing reporter from both sides silences logging on both."""
    legacy, bridge = pubsub_pair
    legacy_reporter = EventReporter(event_source="removal")
    bridge_reporter = EventReporter(event_source="removal")

    event = _DataclassEvent(value=1)

    with patch(_ER_LOGGER_PATH) as mock_log:
        legacy.add_listener(_TAG, legacy_reporter)
        bridge.add_listener(int(_TAG), bridge_reporter)

        # Fire once — both reporters should log (2 calls total).
        legacy.trigger_event(_TAG, event)
        bridge.trigger_event(int(_TAG), event)

        assert mock_log.event_log.call_count == 2, (
            f"Expected 2 calls after first fire, got {mock_log.event_log.call_count}"
        )

        # Remove then fire again — neither reporter should log.
        legacy.remove_listener(_TAG, legacy_reporter)
        bridge.remove_listener(int(_TAG), bridge_reporter)
        legacy.trigger_event(_TAG, event)
        bridge.trigger_event(int(_TAG), event)

    assert mock_log.event_log.call_count == 2, (
        f"Expected still 2 calls after removal, got {mock_log.event_log.call_count}"
    )
