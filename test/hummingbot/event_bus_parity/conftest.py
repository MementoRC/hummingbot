"""Shared fixtures and helpers for the event-bus parity test suite (B5-B14).

PubSub vs PubSubBridge comparisons require consistent test doubles across all
parity modules.  This conftest provides:

- ``RecorderListener``   — EventListener subclass that captures every dispatch.
- ``pubsub_pair``        — pytest fixture: (PubSub, PubSubBridge) standalone pair.
- ``event_recorder``     — pytest fixture: factory returning fresh RecorderListeners.
- ``register_on_both``   — module-level helper that subscribes one listener to both
                           sides while handling the Enum/int API difference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from event_bus import EventBus

from hummingbot.core.event.event_listener import EventListener
from hummingbot.core.event_bus_bridge import PubSubBridge
from hummingbot.core.pubsub import PubSub

if TYPE_CHECKING:
    from collections.abc import Callable
    from enum import Enum


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


class RecorderListener(EventListener):
    """Capture every dispatch for diff against a twin recorder.

    Subclasses ``EventListener`` (Cython) so it is accepted by both PubSub
    (which requires a true EventListener for weak-reference handling) and
    PubSubBridge.

    ``__call__`` is overridden rather than ``c_call`` because the Cython
    ``c_call`` implementation calls ``self(arg)``, so overriding ``__call__``
    is the correct extension point and avoids infinite recursion.
    """

    def __init__(self, name: str = "rec") -> None:
        super().__init__()
        self.name: str = name
        self.calls: list[tuple[str, Any]] = []

    def __call__(self, arg: Any) -> None:  # type: ignore[override]
        self.calls.append((self.name, arg))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pubsub_pair() -> tuple[PubSub, PubSubBridge]:
    """Return ``(legacy_pubsub, bridge)`` — standalone instances with no shared state.

    Use ``register_on_both()`` to add a listener to both sides simultaneously.
    """
    legacy = PubSub()
    bridge = PubSubBridge(EventBus(name="parity-test"))
    return legacy, bridge


@pytest.fixture
def event_recorder() -> Callable[[str], RecorderListener]:
    """Factory fixture: ``event_recorder("name")`` returns a fresh RecorderListener."""

    def _make(name: str = "rec") -> RecorderListener:
        return RecorderListener(name)

    return _make


# ---------------------------------------------------------------------------
# Module-level helper (importable by parity test modules)
# ---------------------------------------------------------------------------


def register_on_both(
    legacy: PubSub,
    bridge: PubSubBridge,
    event_tag: Enum,
    listener: EventListener,
) -> None:
    """Subscribe *listener* to *event_tag* on both *legacy* and *bridge*.

    PubSub accepts an ``Enum`` and calls ``.value`` internally.
    PubSubBridge accepts an ``int`` directly (already the ``.value``).
    This helper handles the difference so callers can pass a single Enum tag.
    """
    legacy.add_listener(event_tag, listener)
    bridge.add_listener(event_tag.value, listener)
