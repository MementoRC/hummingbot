"""B11 — StrategyBase event topology parity tests.

Verifies that the six order lifecycle events fired by StrategyBase
(BuyOrderCreated, SellOrderCreated, OrderFilled, OrderCancelled,
OrderFailure, BuyOrderCompleted, SellOrderCompleted) are delivered
identically by legacy PubSub and PubSubBridge.

Approach: directly call ``trigger_event`` / ``c_trigger_event`` on each
side with a synthetic payload, and assert the RecorderListener on both
sides captures an identical call list — isolates event topology from
connector dependencies.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.trade_fee import AddedToCostTradeFee
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    MarketEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)

from .conftest import register_on_both

if TYPE_CHECKING:
    from .conftest import RecorderListener


# ---------------------------------------------------------------------------
# Helpers — minimal synthetic event payloads
# ---------------------------------------------------------------------------

_TS = 1_700_000_000.0
_ORDER_ID = "order-abc-123"
_TRADING_PAIR = "BTC-USDT"
_BASE = "BTC"
_QUOTE = "USDT"
_PRICE = Decimal("30000")
_AMOUNT = Decimal("0.001")
_FEE = AddedToCostTradeFee()


def _buy_created() -> BuyOrderCreatedEvent:
    return BuyOrderCreatedEvent(
        timestamp=_TS,
        type=OrderType.LIMIT,
        trading_pair=_TRADING_PAIR,
        amount=_AMOUNT,
        price=_PRICE,
        order_id=_ORDER_ID,
        creation_timestamp=_TS,
    )


def _sell_created() -> SellOrderCreatedEvent:
    return SellOrderCreatedEvent(
        timestamp=_TS,
        type=OrderType.LIMIT,
        trading_pair=_TRADING_PAIR,
        amount=_AMOUNT,
        price=_PRICE,
        order_id=_ORDER_ID,
        creation_timestamp=_TS,
    )


def _order_filled() -> OrderFilledEvent:
    return OrderFilledEvent(
        timestamp=_TS,
        order_id=_ORDER_ID,
        trading_pair=_TRADING_PAIR,
        trade_type=TradeType.BUY,
        order_type=OrderType.LIMIT,
        price=_PRICE,
        amount=_AMOUNT,
        trade_fee=_FEE,
    )


def _order_cancelled() -> OrderCancelledEvent:
    return OrderCancelledEvent(timestamp=_TS, order_id=_ORDER_ID)


def _order_failed() -> MarketOrderFailureEvent:
    return MarketOrderFailureEvent(
        timestamp=_TS,
        order_id=_ORDER_ID,
        order_type=OrderType.LIMIT,
    )


def _buy_completed() -> BuyOrderCompletedEvent:
    return BuyOrderCompletedEvent(
        timestamp=_TS,
        order_id=_ORDER_ID,
        base_asset=_BASE,
        quote_asset=_QUOTE,
        base_asset_amount=_AMOUNT,
        quote_asset_amount=_PRICE * _AMOUNT,
        order_type=OrderType.LIMIT,
    )


def _sell_completed() -> SellOrderCompletedEvent:
    return SellOrderCompletedEvent(
        timestamp=_TS,
        order_id=_ORDER_ID,
        base_asset=_BASE,
        quote_asset=_QUOTE,
        base_asset_amount=_AMOUNT,
        quote_asset_amount=_PRICE * _AMOUNT,
        order_type=OrderType.LIMIT,
    )


# ---------------------------------------------------------------------------
# Parity helper
# ---------------------------------------------------------------------------


def _assert_parity(
    legacy_rec: RecorderListener,
    bridge_rec: RecorderListener,
) -> None:
    """Both recorders must have captured exactly one call with identical payload."""
    assert len(legacy_rec.calls) == 1, f"legacy got {len(legacy_rec.calls)} calls, want 1"
    assert len(bridge_rec.calls) == 1, f"bridge got {len(bridge_rec.calls)} calls, want 1"
    _legacy_name, legacy_payload = legacy_rec.calls[0]
    _bridge_name, bridge_payload = bridge_rec.calls[0]
    assert legacy_payload == bridge_payload, (
        f"Payload mismatch:\n  legacy: {legacy_payload!r}\n  bridge: {bridge_payload!r}"
    )


# ---------------------------------------------------------------------------
# Tests — one per event tag that StrategyBase subscribes to
# ---------------------------------------------------------------------------


class TestBuyOrderCreatedParity:
    """MarketEvent.BuyOrderCreated (tag 200) — 'order placed' for buy side."""

    def test_trigger_delivers_to_both(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.BuyOrderCreated, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.BuyOrderCreated, bridge_rec)

        payload = _buy_created()
        legacy.trigger_event(MarketEvent.BuyOrderCreated, payload)
        bridge.trigger_event(MarketEvent.BuyOrderCreated.value, payload)

        _assert_parity(legacy_rec, bridge_rec)

    def test_c_trigger_event_equivalent(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.BuyOrderCreated, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.BuyOrderCreated, bridge_rec)

        payload = _buy_created()
        legacy.trigger_event(MarketEvent.BuyOrderCreated, payload)
        bridge.c_trigger_event(MarketEvent.BuyOrderCreated.value, payload)

        _assert_parity(legacy_rec, bridge_rec)


class TestSellOrderCreatedParity:
    """MarketEvent.SellOrderCreated (tag 201) — 'order placed' for sell side."""

    def test_trigger_delivers_to_both(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.SellOrderCreated, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.SellOrderCreated, bridge_rec)

        payload = _sell_created()
        legacy.trigger_event(MarketEvent.SellOrderCreated, payload)
        bridge.trigger_event(MarketEvent.SellOrderCreated.value, payload)

        _assert_parity(legacy_rec, bridge_rec)

    def test_c_trigger_event_equivalent(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.SellOrderCreated, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.SellOrderCreated, bridge_rec)

        payload = _sell_created()
        legacy.trigger_event(MarketEvent.SellOrderCreated, payload)
        bridge.c_trigger_event(MarketEvent.SellOrderCreated.value, payload)

        _assert_parity(legacy_rec, bridge_rec)


class TestOrderFilledParity:
    """MarketEvent.OrderFilled (tag 107) — 'order filled'."""

    def test_trigger_delivers_to_both(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.OrderFilled, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.OrderFilled, bridge_rec)

        payload = _order_filled()
        legacy.trigger_event(MarketEvent.OrderFilled, payload)
        bridge.trigger_event(MarketEvent.OrderFilled.value, payload)

        _assert_parity(legacy_rec, bridge_rec)

    def test_c_trigger_event_equivalent(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.OrderFilled, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.OrderFilled, bridge_rec)

        payload = _order_filled()
        legacy.trigger_event(MarketEvent.OrderFilled, payload)
        bridge.c_trigger_event(MarketEvent.OrderFilled.value, payload)

        _assert_parity(legacy_rec, bridge_rec)


class TestOrderCancelledParity:
    """MarketEvent.OrderCancelled (tag 106) — 'order cancelled'."""

    def test_trigger_delivers_to_both(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.OrderCancelled, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.OrderCancelled, bridge_rec)

        payload = _order_cancelled()
        legacy.trigger_event(MarketEvent.OrderCancelled, payload)
        bridge.trigger_event(MarketEvent.OrderCancelled.value, payload)

        _assert_parity(legacy_rec, bridge_rec)

    def test_c_trigger_event_equivalent(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.OrderCancelled, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.OrderCancelled, bridge_rec)

        payload = _order_cancelled()
        legacy.trigger_event(MarketEvent.OrderCancelled, payload)
        bridge.c_trigger_event(MarketEvent.OrderCancelled.value, payload)

        _assert_parity(legacy_rec, bridge_rec)


class TestOrderFailureParity:
    """MarketEvent.OrderFailure (tag 198) — 'order failed'."""

    def test_trigger_delivers_to_both(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.OrderFailure, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.OrderFailure, bridge_rec)

        payload = _order_failed()
        legacy.trigger_event(MarketEvent.OrderFailure, payload)
        bridge.trigger_event(MarketEvent.OrderFailure.value, payload)

        _assert_parity(legacy_rec, bridge_rec)

    def test_c_trigger_event_equivalent(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.OrderFailure, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.OrderFailure, bridge_rec)

        payload = _order_failed()
        legacy.trigger_event(MarketEvent.OrderFailure, payload)
        bridge.c_trigger_event(MarketEvent.OrderFailure.value, payload)

        _assert_parity(legacy_rec, bridge_rec)


class TestBuyOrderCompletedParity:
    """MarketEvent.BuyOrderCompleted (tag 102) — 'buy order fully filled'."""

    def test_trigger_delivers_to_both(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.BuyOrderCompleted, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.BuyOrderCompleted, bridge_rec)

        payload = _buy_completed()
        legacy.trigger_event(MarketEvent.BuyOrderCompleted, payload)
        bridge.trigger_event(MarketEvent.BuyOrderCompleted.value, payload)

        _assert_parity(legacy_rec, bridge_rec)

    def test_c_trigger_event_equivalent(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.BuyOrderCompleted, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.BuyOrderCompleted, bridge_rec)

        payload = _buy_completed()
        legacy.trigger_event(MarketEvent.BuyOrderCompleted, payload)
        bridge.c_trigger_event(MarketEvent.BuyOrderCompleted.value, payload)

        _assert_parity(legacy_rec, bridge_rec)


class TestSellOrderCompletedParity:
    """MarketEvent.SellOrderCompleted (tag 103) — 'sell order fully filled'."""

    def test_trigger_delivers_to_both(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.SellOrderCompleted, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.SellOrderCompleted, bridge_rec)

        payload = _sell_completed()
        legacy.trigger_event(MarketEvent.SellOrderCompleted, payload)
        bridge.trigger_event(MarketEvent.SellOrderCompleted.value, payload)

        _assert_parity(legacy_rec, bridge_rec)

    def test_c_trigger_event_equivalent(self, pubsub_pair, event_recorder):
        legacy, bridge = pubsub_pair
        legacy_rec = event_recorder("legacy")
        bridge_rec = event_recorder("bridge")
        register_on_both(legacy, bridge, MarketEvent.SellOrderCompleted, legacy_rec)
        register_on_both(legacy, bridge, MarketEvent.SellOrderCompleted, bridge_rec)

        payload = _sell_completed()
        legacy.trigger_event(MarketEvent.SellOrderCompleted, payload)
        bridge.c_trigger_event(MarketEvent.SellOrderCompleted.value, payload)

        _assert_parity(legacy_rec, bridge_rec)


class TestMultiListenerTopology:
    """Multiple listeners on the same tag — both buses fan-out identically."""

    @pytest.mark.parametrize(
        "event_tag",
        [
            MarketEvent.BuyOrderCreated,
            MarketEvent.OrderFilled,
            MarketEvent.OrderCancelled,
        ],
    )
    def test_fanout_parity(self, pubsub_pair, event_recorder, event_tag):
        legacy, bridge = pubsub_pair
        rec_a = event_recorder("a")
        rec_b = event_recorder("b")
        register_on_both(legacy, bridge, event_tag, rec_a)
        register_on_both(legacy, bridge, event_tag, rec_b)

        if event_tag == MarketEvent.BuyOrderCreated:
            payload = _buy_created()
        elif event_tag == MarketEvent.OrderFilled:
            payload = _order_filled()
        else:
            payload = _order_cancelled()

        legacy.trigger_event(event_tag, payload)
        bridge.trigger_event(event_tag.value, payload)

        # Both listeners must have received exactly one call on each side.
        assert len(rec_a.calls) == 2, f"rec_a got {len(rec_a.calls)}, want 2"
        assert len(rec_b.calls) == 2, f"rec_b got {len(rec_b.calls)}, want 2"
        # Payloads are identical across both deliveries.
        assert rec_a.calls[0][1] == rec_a.calls[1][1]
        assert rec_b.calls[0][1] == rec_b.calls[1][1]

    def test_deregister_stops_delivery(self, pubsub_pair, event_recorder):
        """Removing a listener on both sides stops further delivery."""
        legacy, bridge = pubsub_pair
        rec = event_recorder("rec")
        register_on_both(legacy, bridge, MarketEvent.OrderFilled, rec)

        legacy.remove_listener(MarketEvent.OrderFilled, rec)
        bridge.remove_listener(MarketEvent.OrderFilled.value, rec)

        legacy.trigger_event(MarketEvent.OrderFilled, _order_filled())
        bridge.trigger_event(MarketEvent.OrderFilled.value, _order_filled())

        assert rec.calls == [], f"Expected no calls after removal, got {rec.calls}"
