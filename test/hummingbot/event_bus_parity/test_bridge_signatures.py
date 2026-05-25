"""Signature parity tests: PubSubBridge vs legacy PubSub.

Phase B1 gate — verifies that the bridge exposes every Python-level public
method that legacy PubSub exposes, with the same parameter names (order and
names, not annotations) so call sites can be mechanically substituted.

Note on event_tag type: legacy PubSub accepts ``Enum`` and internally calls
``.value`` before forwarding to the C-level.  PubSubBridge accepts ``int``
directly (already the ``.value``).  The parameter *name* is preserved; the
annotation diverges intentionally and is not compared here.
"""

from __future__ import annotations

import inspect

import pytest

from hummingbot.core.event_bus_bridge import PubSubBridge
from hummingbot.core.pubsub import PubSub

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _param_names(method: object) -> list[str]:
    """Return non-self parameter names for a method."""
    sig = inspect.signature(method)
    keys = list(sig.parameters.keys())
    # drop 'self' (first positional on unbound methods)
    return keys[1:] if keys and keys[0] == "self" else keys


# ---------------------------------------------------------------------------
# Core signature parity
# ---------------------------------------------------------------------------

LEGACY_METHODS = [
    "add_listener",
    "remove_listener",
    "get_listeners",
    "trigger_event",
]


@pytest.mark.parametrize("method_name", LEGACY_METHODS)
def test_bridge_has_method(method_name: str) -> None:
    assert hasattr(PubSubBridge, method_name), f"PubSubBridge missing method: {method_name}"


@pytest.mark.parametrize("method_name", LEGACY_METHODS)
def test_bridge_method_param_names_match_legacy(method_name: str) -> None:
    bridge_method = getattr(PubSubBridge, method_name)
    pubsub_method = getattr(PubSub, method_name)

    bridge_params = _param_names(bridge_method)
    pubsub_params = _param_names(pubsub_method)

    assert bridge_params == pubsub_params, (
        f"Parameter name mismatch for {method_name}: bridge={bridge_params!r} legacy={pubsub_params!r}"
    )


# ---------------------------------------------------------------------------
# c_trigger_event — Cython-callable variant present on bridge
# ---------------------------------------------------------------------------


def test_bridge_has_c_trigger_event() -> None:
    assert hasattr(PubSubBridge, "c_trigger_event"), "PubSubBridge missing c_trigger_event (Cython-callable variant)"


def test_c_trigger_event_param_names_match_trigger_event() -> None:
    """c_trigger_event must accept the same args as trigger_event."""
    bridge_trigger = _param_names(PubSubBridge.trigger_event)
    bridge_c_trigger = _param_names(PubSubBridge.c_trigger_event)
    assert bridge_trigger == bridge_c_trigger, (
        f"trigger_event params {bridge_trigger!r} != c_trigger_event params {bridge_c_trigger!r}"
    )


# ---------------------------------------------------------------------------
# Instantiation smoke test
# ---------------------------------------------------------------------------


def test_pubsubbridge_instantiates() -> None:
    bridge = PubSubBridge()
    assert bridge is not None


def test_pubsubbridge_instantiates_with_custom_prefix() -> None:
    bridge = PubSubBridge(name_prefix="executor")
    assert bridge is not None
