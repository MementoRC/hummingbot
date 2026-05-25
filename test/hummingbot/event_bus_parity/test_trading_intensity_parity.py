"""Parity tests — trading_intensity indicator event topology (Plan task B14).

Verifies that ``PubSubBridge`` and legacy ``PubSub`` produce identical
observable behaviour for the event subscription pattern used by
``TradingIntensityIndicator``:

* The indicator registers a ``TradesForwarder`` (EventListener subclass) on
  ``OrderBookEvent.TradeEvent`` when it is initialised.
* Trades arrive via ``trigger_event`` / ``c_trigger_event``.
* The listener receives each trade payload in registration order.

These tests drive ``PubSub`` and ``PubSubBridge`` in isolation — they do
**not** construct a live ``TradingIntensityIndicator`` (which requires Cython
compilation and a real ``OrderBook``).  Instead they replicate the exact
subscription pattern so that topology differences surface here, not in the
production indicator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from hummingbot.core.event.events import OrderBookEvent
from hummingbot.core.event_bus_bridge import PubSubBridge
from hummingbot.core.pubsub import PubSub

from .conftest import RecorderListener

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRADE_TAG = OrderBookEvent.TradeEvent  # IntEnum value 901


def _make_trade_event(price: float = 100.0, amount: float = 1.0) -> MagicMock:
    """Return a lightweight trade-event stand-in (matches what c_register_trade receives)."""
    evt = MagicMock()
    evt.price = price
    evt.amount = amount
    evt.timestamp = 1_000_000.0
    return evt


# ---------------------------------------------------------------------------
# B14-T1: add_listener — listener appears in get_listeners for both buses
# ---------------------------------------------------------------------------


def test_add_listener_appears_in_get_listeners_legacy() -> None:
    """Legacy PubSub: listener registered for TradeEvent is returned by get_listeners."""
    legacy = PubSub()
    rec = RecorderListener("legacy-reg")
    legacy.add_listener(_TRADE_TAG, rec)
    assert rec in legacy.get_listeners(_TRADE_TAG)


def test_add_listener_appears_in_get_listeners_bridge() -> None:
    """PubSubBridge: listener registered for TradeEvent is returned by get_listeners."""
    from event_bus import EventBus

    bridge = PubSubBridge(EventBus(name="b14-t1"))
    rec = RecorderListener("bridge-reg")
    bridge.add_listener(_TRADE_TAG.value, rec)
    assert rec in bridge.get_listeners(_TRADE_TAG.value)


# ---------------------------------------------------------------------------
# B14-T2: trigger_event dispatches to registered listener — topology parity
# ---------------------------------------------------------------------------


def test_trigger_event_dispatches_to_listener_parity(pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
    """trigger_event reaches the registered listener on both buses identically.

    Each recorder is registered on its own side only so that triggering on one
    side does not produce spurious calls on the other recorder.
    """
    legacy, bridge = pubsub_pair
    rec_leg = RecorderListener("leg")
    rec_br = RecorderListener("br")
    # Isolated registration: each recorder sees only its own bus triggers.
    legacy.add_listener(_TRADE_TAG, rec_leg)
    bridge.add_listener(_TRADE_TAG.value, rec_br)

    trade = _make_trade_event()
    legacy.trigger_event(_TRADE_TAG, trade)
    bridge.trigger_event(_TRADE_TAG.value, trade)

    assert len(rec_leg.calls) == 1, "legacy listener must receive exactly one dispatch"
    assert len(rec_br.calls) == 1, "bridge listener must receive exactly one dispatch"
    assert rec_leg.calls[0][1] is trade
    assert rec_br.calls[0][1] is trade


# ---------------------------------------------------------------------------
# B14-T3: c_trigger_event is equivalent to trigger_event on the bridge
# ---------------------------------------------------------------------------


def test_c_trigger_event_equivalent_to_trigger_event_bridge() -> None:
    """PubSubBridge.c_trigger_event and trigger_event must dispatch identically."""
    from event_bus import EventBus

    bridge_a = PubSubBridge(EventBus(name="b14-t3a"))
    bridge_b = PubSubBridge(EventBus(name="b14-t3b"))

    rec_a = RecorderListener("via-trigger")
    rec_b = RecorderListener("via-c-trigger")
    bridge_a.add_listener(_TRADE_TAG.value, rec_a)
    bridge_b.add_listener(_TRADE_TAG.value, rec_b)

    trade = _make_trade_event(price=200.0)
    bridge_a.trigger_event(_TRADE_TAG.value, trade)
    bridge_b.c_trigger_event(_TRADE_TAG.value, trade)

    assert rec_a.calls == [("via-trigger", trade)]
    assert rec_b.calls == [("via-c-trigger", trade)]


# ---------------------------------------------------------------------------
# B14-T4: multiple trades dispatched in order — parity
# ---------------------------------------------------------------------------


def test_multiple_trades_received_in_order_parity(pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
    """Listener receives multiple trades in dispatch order on both buses."""
    legacy, bridge = pubsub_pair
    rec_leg = RecorderListener("leg-multi")
    rec_br = RecorderListener("br-multi")

    legacy.add_listener(_TRADE_TAG, rec_leg)
    bridge.add_listener(_TRADE_TAG.value, rec_br)

    trades = [_make_trade_event(price=float(p)) for p in (100, 101, 102)]
    for t in trades:
        legacy.trigger_event(_TRADE_TAG, t)
        bridge.trigger_event(_TRADE_TAG.value, t)

    assert [c[1] for c in rec_leg.calls] == trades, "legacy must receive trades in order"
    assert [c[1] for c in rec_br.calls] == trades, "bridge must receive trades in order"


# ---------------------------------------------------------------------------
# B14-T5: remove_listener stops future dispatches — parity
# ---------------------------------------------------------------------------


def test_remove_listener_stops_dispatches_legacy() -> None:
    """Legacy PubSub: removed listener receives no further events."""
    legacy = PubSub()
    rec = RecorderListener("leg-remove")
    legacy.add_listener(_TRADE_TAG, rec)
    legacy.remove_listener(_TRADE_TAG, rec)
    legacy.trigger_event(_TRADE_TAG, _make_trade_event())
    assert rec.calls == [], "removed listener must not receive any events"


def test_remove_listener_stops_dispatches_bridge() -> None:
    """PubSubBridge: removed listener receives no further events."""
    from event_bus import EventBus

    bridge = PubSubBridge(EventBus(name="b14-t5"))
    rec = RecorderListener("br-remove")
    bridge.add_listener(_TRADE_TAG.value, rec)
    bridge.remove_listener(_TRADE_TAG.value, rec)
    bridge.trigger_event(_TRADE_TAG.value, _make_trade_event())
    assert rec.calls == [], "removed listener must not receive any events"


# ---------------------------------------------------------------------------
# B14-T6: remove_listener idempotent — no error on double-remove
# ---------------------------------------------------------------------------


def test_remove_listener_idempotent_legacy() -> None:
    """Legacy PubSub: double remove_listener must not raise."""
    legacy = PubSub()
    rec = RecorderListener("leg-idem")
    legacy.add_listener(_TRADE_TAG, rec)
    legacy.remove_listener(_TRADE_TAG, rec)
    legacy.remove_listener(_TRADE_TAG, rec)  # must not raise


def test_remove_listener_idempotent_bridge() -> None:
    """PubSubBridge: double remove_listener must not raise."""
    from event_bus import EventBus

    bridge = PubSubBridge(EventBus(name="b14-t6"))
    rec = RecorderListener("br-idem")
    bridge.add_listener(_TRADE_TAG.value, rec)
    bridge.remove_listener(_TRADE_TAG.value, rec)
    bridge.remove_listener(_TRADE_TAG.value, rec)  # must not raise


# ---------------------------------------------------------------------------
# B14-T7: duplicate add_listener is a no-op — listener fires once only
# ---------------------------------------------------------------------------


def test_duplicate_add_listener_noop_legacy() -> None:
    """Legacy PubSub: registering the same listener twice fires it only once per event."""
    legacy = PubSub()
    rec = RecorderListener("leg-dup")
    legacy.add_listener(_TRADE_TAG, rec)
    legacy.add_listener(_TRADE_TAG, rec)
    legacy.trigger_event(_TRADE_TAG, _make_trade_event())
    assert len(rec.calls) == 1, "duplicate registration must not cause double-dispatch (legacy)"


def test_duplicate_add_listener_noop_bridge() -> None:
    """PubSubBridge: registering the same listener twice fires it only once per event."""
    from event_bus import EventBus

    bridge = PubSubBridge(EventBus(name="b14-t7"))
    rec = RecorderListener("br-dup")
    bridge.add_listener(_TRADE_TAG.value, rec)
    bridge.add_listener(_TRADE_TAG.value, rec)
    bridge.trigger_event(_TRADE_TAG.value, _make_trade_event())
    assert len(rec.calls) == 1, "duplicate registration must not cause double-dispatch (bridge)"


# ---------------------------------------------------------------------------
# B14-T8: unregistered event tag produces no dispatch — parity
# ---------------------------------------------------------------------------


def test_unregistered_tag_no_dispatch_parity(pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
    """Triggering a tag that has no listener must produce zero calls on both buses."""
    legacy, bridge = pubsub_pair
    rec_leg = RecorderListener("leg-unreg")
    rec_br = RecorderListener("br-unreg")

    # Register on a *different* tag so listeners exist but not for TradeEvent
    other_tag = OrderBookEvent.OrderBookDataSourceUpdateEvent  # 904
    legacy.add_listener(other_tag, rec_leg)
    bridge.add_listener(other_tag.value, rec_br)

    # Trigger the TradeEvent — neither listener should fire
    legacy.trigger_event(_TRADE_TAG, _make_trade_event())
    bridge.trigger_event(_TRADE_TAG.value, _make_trade_event())

    assert rec_leg.calls == [], "legacy must not dispatch to listener on different tag"
    assert rec_br.calls == [], "bridge must not dispatch to listener on different tag"


# ---------------------------------------------------------------------------
# B14-T9: payload identity preserved across bridge callback closure
# ---------------------------------------------------------------------------


def test_payload_identity_preserved_bridge() -> None:
    """Bridge callback closure must pass the exact payload object, not a copy."""
    from event_bus import EventBus

    bridge = PubSubBridge(EventBus(name="b14-t9"))
    rec = RecorderListener("identity")
    bridge.add_listener(_TRADE_TAG.value, rec)

    trade = _make_trade_event()
    bridge.trigger_event(_TRADE_TAG.value, trade)

    assert rec.calls[0][1] is trade, "bridge must preserve payload object identity"
