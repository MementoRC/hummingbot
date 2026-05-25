"""B9 — OrderBook event topology parity tests.

Verifies that ``OrderBook.c_trigger_event`` fires events with identical
topology (call count, event tag, payload identity) when listeners are
attached via the legacy PubSub path and via PubSubBridge.

Events covered:
- OrderBookEvent.TradeEvent (tag 901) — fired by ``apply_trade``
- OrderBookEvent.OrderBookDataSourceUpdateEvent (tag 904) — fired directly
  via ``c_trigger_event`` (as would happen from a data-source layer)

Design notes:
- OrderBook inherits PubSub, so ``add_listener`` / ``c_trigger_event`` on
  the instance IS the legacy path.
- The bridge path is tested by wiring a separate PubSubBridge to the same
  event tags and dispatching equivalent events through it.
- Both paths are driven independently (OrderBook for legacy, PubSubBridge
  standalone for bridge) to isolate topology differences from object-sharing.
"""

from __future__ import annotations

from decimal import Decimal
from test.hummingbot.event_bus_parity.conftest import RecorderListener, register_on_both
from typing import TYPE_CHECKING

from hummingbot.core.data_type.common import TradeType
from hummingbot.core.event.events import OrderBookEvent, OrderBookTradeEvent

if TYPE_CHECKING:
    from hummingbot.core.event_bus_bridge import PubSubBridge
    from hummingbot.core.pubsub import PubSub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade_event(price: float = 100.0, amount: float = 1.0) -> OrderBookTradeEvent:
    """Return a minimal OrderBookTradeEvent for triggering the TradeEvent tag."""
    return OrderBookTradeEvent(
        trading_pair="BTC-USDT",
        timestamp=1_700_000_000.0,
        type=TradeType.BUY,
        price=Decimal(str(price)),
        amount=Decimal(str(amount)),
    )


def _make_orderbook() -> object:
    """Import and instantiate OrderBook (Cython extension)."""
    from hummingbot.core.data_type.order_book import OrderBook

    return OrderBook()


# ---------------------------------------------------------------------------
# TradeEvent (tag 901) topology parity
# ---------------------------------------------------------------------------


class TestTradeEventParity:
    """OrderBook.TradeEvent fires once per apply_trade call on both paths."""

    def test_trade_event_fires_once_on_legacy_path(self) -> None:
        """Legacy: one apply_trade → one TradeEvent dispatch to the listener."""
        ob = _make_orderbook()
        rec = RecorderListener("legacy-trade")
        ob.add_listener(OrderBookEvent.TradeEvent, rec)

        ob.apply_trade(_make_trade_event())

        assert len(rec.calls) == 1

    def test_trade_event_fires_once_on_bridge_path(self, pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
        """Bridge: one trigger_event → one TradeEvent dispatch to the listener."""
        _legacy, bridge = pubsub_pair
        rec = RecorderListener("bridge-trade")
        bridge.add_listener(OrderBookEvent.TradeEvent.value, rec)

        bridge.trigger_event(OrderBookEvent.TradeEvent.value, _make_trade_event())

        assert len(rec.calls) == 1

    def test_trade_event_call_counts_are_identical(self, pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
        """Both paths produce the same call count for N apply/trigger calls."""
        _legacy, bridge = pubsub_pair
        ob = _make_orderbook()

        rec_legacy = RecorderListener("legacy")
        rec_bridge = RecorderListener("bridge")

        ob.add_listener(OrderBookEvent.TradeEvent, rec_legacy)
        bridge.add_listener(OrderBookEvent.TradeEvent.value, rec_bridge)

        n = 3
        trade = _make_trade_event()
        for _ in range(n):
            ob.apply_trade(trade)
            bridge.trigger_event(OrderBookEvent.TradeEvent.value, trade)

        assert len(rec_legacy.calls) == n
        assert len(rec_bridge.calls) == n

    def test_trade_event_payload_identity_preserved(self) -> None:
        """Legacy path: the exact payload object passed in is received by the listener."""
        ob = _make_orderbook()
        rec = RecorderListener("payload-check")
        ob.add_listener(OrderBookEvent.TradeEvent, rec)

        trade = _make_trade_event(price=42.0, amount=0.5)
        ob.apply_trade(trade)

        _name, received = rec.calls[0]
        assert received is trade

    def test_trade_event_payload_identity_bridge(self, pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
        """Bridge path: the exact payload object is forwarded unchanged."""
        _legacy, bridge = pubsub_pair
        rec = RecorderListener("bridge-payload")
        bridge.add_listener(OrderBookEvent.TradeEvent.value, rec)

        trade = _make_trade_event(price=99.0, amount=2.0)
        bridge.trigger_event(OrderBookEvent.TradeEvent.value, trade)

        _name, received = rec.calls[0]
        assert received is trade

    def test_trade_event_no_cross_tag_bleed(self, pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
        """A listener on TradeEvent must not fire when OrderBookDataSourceUpdateEvent fires."""
        _legacy, bridge = pubsub_pair
        trade_rec = RecorderListener("trade-only")
        bridge.add_listener(OrderBookEvent.TradeEvent.value, trade_rec)

        # Fire the OTHER tag — trade_rec must stay silent
        bridge.trigger_event(OrderBookEvent.OrderBookDataSourceUpdateEvent.value, {"source": "diff"})

        assert len(trade_rec.calls) == 0


# ---------------------------------------------------------------------------
# OrderBookDataSourceUpdateEvent (tag 904) topology parity
# ---------------------------------------------------------------------------


class TestOrderBookDataSourceUpdateEventParity:
    """OrderBookDataSourceUpdateEvent fires correctly on both paths."""

    def test_datasource_update_fires_once_legacy(self) -> None:
        """Legacy: direct trigger_event on an OrderBook reaches the listener.

        OrderBook.c_trigger_event is a Cython cdef method — not Python-callable.
        trigger_event is the Python-accessible equivalent.
        PubSub.trigger_event expects an Enum (calls .value internally).
        """
        ob = _make_orderbook()
        rec = RecorderListener("legacy-ds")
        ob.add_listener(OrderBookEvent.OrderBookDataSourceUpdateEvent, rec)

        payload = {"source": "diff", "uid": 1}
        ob.trigger_event(OrderBookEvent.OrderBookDataSourceUpdateEvent, payload)

        assert len(rec.calls) == 1

    def test_datasource_update_fires_once_bridge(self, pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
        """Bridge: trigger_event for tag 904 reaches the listener exactly once."""
        _legacy, bridge = pubsub_pair
        rec = RecorderListener("bridge-ds")
        bridge.add_listener(OrderBookEvent.OrderBookDataSourceUpdateEvent.value, rec)

        payload = {"source": "snapshot", "uid": 2}
        bridge.trigger_event(OrderBookEvent.OrderBookDataSourceUpdateEvent.value, payload)

        assert len(rec.calls) == 1

    def test_datasource_update_call_counts_match(self, pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
        """N dispatches on legacy and bridge both produce N calls."""
        _legacy, bridge = pubsub_pair
        ob = _make_orderbook()

        rec_legacy = RecorderListener("leg-ds")
        rec_bridge = RecorderListener("br-ds")

        ob.add_listener(OrderBookEvent.OrderBookDataSourceUpdateEvent, rec_legacy)
        bridge.add_listener(OrderBookEvent.OrderBookDataSourceUpdateEvent.value, rec_bridge)

        tag_enum = OrderBookEvent.OrderBookDataSourceUpdateEvent
        tag_int = tag_enum.value
        n = 4
        for i in range(n):
            payload = {"uid": i}
            ob.trigger_event(tag_enum, payload)
            bridge.trigger_event(tag_int, payload)

        assert len(rec_legacy.calls) == n
        assert len(rec_bridge.calls) == n

    def test_datasource_update_payload_forwarded_legacy(self) -> None:
        """Legacy trigger_event forwards the exact payload object."""
        ob = _make_orderbook()
        rec = RecorderListener("ds-payload")
        ob.add_listener(OrderBookEvent.OrderBookDataSourceUpdateEvent, rec)

        payload = {"source": "rest", "uid": 99}
        ob.trigger_event(OrderBookEvent.OrderBookDataSourceUpdateEvent, payload)

        _name, received = rec.calls[0]
        assert received is payload

    def test_datasource_update_payload_forwarded_bridge(self, pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
        """Bridge trigger_event forwards the exact payload object."""
        _legacy, bridge = pubsub_pair
        rec = RecorderListener("br-ds-payload")
        bridge.add_listener(OrderBookEvent.OrderBookDataSourceUpdateEvent.value, rec)

        payload = {"source": "ws", "uid": 7}
        bridge.trigger_event(OrderBookEvent.OrderBookDataSourceUpdateEvent.value, payload)

        _name, received = rec.calls[0]
        assert received is payload

    def test_datasource_update_no_cross_tag_bleed(self, pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
        """A listener on OrderBookDataSourceUpdateEvent is not triggered by TradeEvent."""
        _legacy, bridge = pubsub_pair
        ds_rec = RecorderListener("ds-only")
        bridge.add_listener(OrderBookEvent.OrderBookDataSourceUpdateEvent.value, ds_rec)

        # Fire the OTHER tag — ds_rec must stay silent
        bridge.trigger_event(OrderBookEvent.TradeEvent.value, _make_trade_event())

        assert len(ds_rec.calls) == 0


# ---------------------------------------------------------------------------
# Cross-path symmetry — register_on_both helper
# ---------------------------------------------------------------------------


class TestCrossPathSymmetry:
    """register_on_both helper produces identical listener behaviour on both sides."""

    def test_register_on_both_trade_event(
        self, pubsub_pair: tuple[PubSub, PubSubBridge], event_recorder: object
    ) -> None:
        """register_on_both wires the same listener; one trigger_event reaches it once."""
        legacy, bridge = pubsub_pair
        rec = RecorderListener("both-trade")
        register_on_both(legacy, bridge, OrderBookEvent.TradeEvent, rec)

        # Trigger on legacy side
        legacy.trigger_event(OrderBookEvent.TradeEvent, _make_trade_event())
        assert len(rec.calls) == 1

        # Trigger on bridge side
        bridge.trigger_event(OrderBookEvent.TradeEvent.value, _make_trade_event())
        assert len(rec.calls) == 2

    def test_register_on_both_datasource_update(self, pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
        """register_on_both for tag 904 delivers one call per side per publish."""
        legacy, bridge = pubsub_pair
        rec = RecorderListener("both-ds")
        register_on_both(legacy, bridge, OrderBookEvent.OrderBookDataSourceUpdateEvent, rec)

        payload = {"uid": 10}
        legacy.trigger_event(OrderBookEvent.OrderBookDataSourceUpdateEvent, payload)
        bridge.trigger_event(OrderBookEvent.OrderBookDataSourceUpdateEvent.value, payload)

        assert len(rec.calls) == 2

    def test_two_tags_independent_via_register_on_both(self, pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
        """Listeners registered for different tags via register_on_both remain isolated."""
        legacy, bridge = pubsub_pair
        trade_rec = RecorderListener("cross-trade")
        ds_rec = RecorderListener("cross-ds")

        register_on_both(legacy, bridge, OrderBookEvent.TradeEvent, trade_rec)
        register_on_both(legacy, bridge, OrderBookEvent.OrderBookDataSourceUpdateEvent, ds_rec)

        bridge.trigger_event(OrderBookEvent.TradeEvent.value, _make_trade_event())

        assert len(trade_rec.calls) == 1
        assert len(ds_rec.calls) == 0

    def test_remove_listener_stops_dispatch_on_bridge(self, pubsub_pair: tuple[PubSub, PubSubBridge]) -> None:
        """remove_listener on bridge prevents further dispatch after unsubscribe."""
        _legacy, bridge = pubsub_pair
        rec = RecorderListener("removable")
        tag_value = OrderBookEvent.TradeEvent.value
        bridge.add_listener(tag_value, rec)

        bridge.trigger_event(tag_value, _make_trade_event())
        assert len(rec.calls) == 1

        bridge.remove_listener(tag_value, rec)
        bridge.trigger_event(tag_value, _make_trade_event())
        assert len(rec.calls) == 1  # no new call after removal

    def test_remove_listener_stops_dispatch_on_legacy(self) -> None:
        """remove_listener on legacy PubSub prevents further dispatch after unsubscribe."""
        ob = _make_orderbook()
        rec = RecorderListener("removable-legacy")
        ob.add_listener(OrderBookEvent.TradeEvent, rec)

        ob.apply_trade(_make_trade_event())
        assert len(rec.calls) == 1

        ob.remove_listener(OrderBookEvent.TradeEvent, rec)
        ob.apply_trade(_make_trade_event())
        assert len(rec.calls) == 1  # no new call after removal
