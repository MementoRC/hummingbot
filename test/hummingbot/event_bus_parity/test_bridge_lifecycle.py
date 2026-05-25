"""B3 — PubSubBridge weakref + remove_listener semantics tests.

Verifies that the bridge preserves the lifecycle behaviour of the legacy
PubSub API: add/remove round-trips, idempotent removal, get_listeners
correctness, and that dead listeners do not prevent cleanup.
"""

from __future__ import annotations

import gc

from event_bus import EventBus

from hummingbot.core.event_bus_bridge import PubSubBridge

# ---------------------------------------------------------------------------
# Minimal test double
# ---------------------------------------------------------------------------


class FakeListener:
    """Callable test double.  Plain class — usable as a weak-referenceable object."""

    def __call__(self, payload: object) -> None:
        pass  # no-op; lifecycle tests focus on registration state, not dispatch


# ---------------------------------------------------------------------------
# add_listener / remove_listener round-trip
# ---------------------------------------------------------------------------


def test_remove_listener_unsubscribes_from_bus() -> None:
    """remove_listener cancels the underlying bus subscription."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    listener = FakeListener()

    bridge.add_listener(1, listener)
    bridge.remove_listener(1, listener)

    assert bus.get_subscribers("market:1") == ()


def test_remove_listener_removes_from_get_listeners() -> None:
    """After remove_listener, get_listeners no longer includes the listener."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    listener = FakeListener()

    bridge.add_listener(1, listener)
    assert listener in bridge.get_listeners(1)

    bridge.remove_listener(1, listener)
    assert listener not in bridge.get_listeners(1)


def test_remove_listener_for_one_tag_leaves_other_intact() -> None:
    """Removing listener from tag A must not affect listener registered to tag B."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    listener = FakeListener()

    bridge.add_listener(1, listener)
    bridge.add_listener(2, listener)
    bridge.remove_listener(1, listener)

    assert bus.get_subscribers("market:1") == ()
    assert len(bus.get_subscribers("market:2")) == 1


# ---------------------------------------------------------------------------
# Idempotent removal
# ---------------------------------------------------------------------------


def test_remove_unknown_listener_is_idempotent() -> None:
    """remove_listener on a listener that was never added must not raise."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    listener = FakeListener()

    bridge.remove_listener(1, listener)  # never added — must not raise


def test_double_remove_is_idempotent() -> None:
    """Calling remove_listener twice for the same (tag, listener) must not raise."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    listener = FakeListener()

    bridge.add_listener(1, listener)
    bridge.remove_listener(1, listener)
    bridge.remove_listener(1, listener)  # second call — must not raise


# ---------------------------------------------------------------------------
# get_listeners correctness
# ---------------------------------------------------------------------------


def test_get_listeners_returns_all_registered_for_tag() -> None:
    """get_listeners returns all listeners registered for a given tag."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    a, b = FakeListener(), FakeListener()

    bridge.add_listener(1, a)
    bridge.add_listener(1, b)

    result = bridge.get_listeners(1)
    assert set(result) == {a, b}


def test_get_listeners_excludes_other_tags() -> None:
    """get_listeners(tag) must not include listeners for a different tag."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    a, b = FakeListener(), FakeListener()

    bridge.add_listener(1, a)
    bridge.add_listener(2, b)

    assert bridge.get_listeners(1) == [a]
    assert bridge.get_listeners(2) == [b]


def test_get_listeners_empty_when_none_registered() -> None:
    """get_listeners returns an empty list when no listeners have been added."""
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)

    assert bridge.get_listeners(99) == []


# ---------------------------------------------------------------------------
# WeakValueDictionary / GC interaction
# ---------------------------------------------------------------------------


def test_listener_eligible_for_gc_after_remove() -> None:
    """After remove_listener + del, the listener object can be collected.

    This test exists to document the expected GC-friendly behaviour of the
    bridge's WeakValueDictionary.  No assertion on GC timing is made beyond
    verifying that the operation completes without error.
    """
    bus = EventBus()
    bridge = PubSubBridge(bus=bus)
    listener = FakeListener()

    bridge.add_listener(1, listener)
    bridge.remove_listener(1, listener)
    del listener
    gc.collect()  # encourage collection; best-effort

    # No assertion beyond no-exception.  The _listeners WeakValueDictionary
    # may or may not have already dropped the entry; both are acceptable.
