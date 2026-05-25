"""B2 — PubSubBridge int tag → str event_type translation tests.

Verifies that add_listener, trigger_event, and the underlying bus subscription
are all keyed by the correct translated topic (``{prefix}:{tag}``), and that
custom prefixes are respected.
"""

from __future__ import annotations

from event_bus import EventBus

from hummingbot.core.event_bus_bridge import PubSubBridge

# ---------------------------------------------------------------------------
# Minimal test double — no EventListener dependency needed at runtime
# ---------------------------------------------------------------------------


class FakeListener:
    """Callable test double that records the last payload it received."""

    def __init__(self) -> None:
        self.called_with: object = None

    def __call__(self, payload: object) -> None:
        self.called_with = payload


# ---------------------------------------------------------------------------
# Topic translation — subscription visible on the underlying bus
# ---------------------------------------------------------------------------


def test_default_prefix_is_market() -> None:
    """add_listener with default prefix registers topic 'market:<tag>' on bus."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    listener = FakeListener()

    bridge.add_listener(101, listener)

    assert len(bus.get_subscribers("market:101")) == 1


def test_custom_prefix_used_for_topic() -> None:
    """add_listener with custom prefix registers topic '<prefix>:<tag>' on bus."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus, name_prefix="executor")
    listener = FakeListener()

    bridge.add_listener(5, listener)

    assert len(bus.get_subscribers("executor:5")) == 1


def test_different_tags_produce_different_topics() -> None:
    """Two distinct int tags map to two distinct topics (no collision)."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    a, b = FakeListener(), FakeListener()

    bridge.add_listener(10, a)
    bridge.add_listener(20, b)

    assert len(bus.get_subscribers("market:10")) == 1
    assert len(bus.get_subscribers("market:20")) == 1
    assert len(bus.get_subscribers("market:10")) != len(bus.get_subscribers("market:99"))


def test_same_tag_not_subscribed_twice_for_same_listener() -> None:
    """Duplicate add_listener calls for the same (tag, listener) are no-ops."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    listener = FakeListener()

    bridge.add_listener(7, listener)
    bridge.add_listener(7, listener)  # duplicate — should be a no-op

    assert len(bus.get_subscribers("market:7")) == 1


# ---------------------------------------------------------------------------
# Dispatch — trigger_event routes through translated topic
# ---------------------------------------------------------------------------


def test_trigger_event_dispatches_to_translated_topic() -> None:
    """trigger_event publishes to the translated topic and listener receives payload."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    listener = FakeListener()

    bridge.add_listener(7, listener)
    bridge.trigger_event(7, "payload")

    assert listener.called_with == "payload"


def test_trigger_event_only_reaches_matching_tag_listener() -> None:
    """trigger_event(tag) must not invoke listeners registered for a different tag."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    a, b = FakeListener(), FakeListener()

    bridge.add_listener(1, a)
    bridge.add_listener(2, b)
    bridge.trigger_event(1, "for_a")

    assert a.called_with == "for_a"
    assert b.called_with is None  # must not have been triggered


def test_c_trigger_event_behaves_identically_to_trigger_event() -> None:
    """c_trigger_event is an alias — same translation and dispatch path."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    listener = FakeListener()

    bridge.add_listener(3, listener)
    bridge.c_trigger_event(3, "via_c")

    assert listener.called_with == "via_c"


def test_translation_is_deterministic() -> None:
    """The same int tag always maps to the same str topic across two bridge instances."""
    bus1, bus2 = EventBus(), EventBus()
    bridge1 = PubSubBridge(bus=bus1, name_prefix="market")
    bridge2 = PubSubBridge(bus=bus2, name_prefix="market")

    l1, l2 = FakeListener(), FakeListener()
    bridge1.add_listener(42, l1)
    bridge2.add_listener(42, l2)

    # Both subscribe to the same derived topic name
    subs1 = bus1.get_subscribers("market:42")
    subs2 = bus2.get_subscribers("market:42")
    assert len(subs1) == 1
    assert len(subs2) == 1
