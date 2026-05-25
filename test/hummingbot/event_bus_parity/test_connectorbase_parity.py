"""Parity tests — ConnectorBase event topology (Plan task B12).

Verifies that PubSubBridge dispatches the 15 MarketEvent tags declared in
ConnectorBase.MARKET_EVENTS identically to legacy PubSub when driven via
``c_trigger_event``.

No real ConnectorBase is instantiated; PubSub and PubSubBridge are driven
directly so the suite stays pure-Python and dependency-free.
"""

from __future__ import annotations

from test.hummingbot.event_bus_parity.conftest import register_on_both
from typing import TYPE_CHECKING

from hummingbot.core.event.events import MarketEvent
from hummingbot.core.pubsub import PubSub

if TYPE_CHECKING:
    from hummingbot.core.event_bus_bridge import PubSubBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _fire_both(legacy: PubSub, bridge: PubSubBridge, tag: MarketEvent, payload: object) -> None:
    """Trigger *tag* with *payload* on both sides.

    PubSub.c_trigger_event is a Cython cdef method — not Python-callable.
    Use trigger_event on the legacy side; bridge supports both but c_trigger_event
    is used here for symmetry with what Cython code would call.

    PubSub.trigger_event expects an Enum (calls .value internally).
    PubSubBridge.c_trigger_event expects an int.
    """
    legacy.trigger_event(tag, payload)
    bridge.c_trigger_event(tag.value, payload)


# ---------------------------------------------------------------------------
# Order-creation events
# ---------------------------------------------------------------------------


def test_buy_order_created_parity(pubsub_pair, event_recorder) -> None:
    """BuyOrderCreated fires to one listener on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("buy_created")
    register_on_both(legacy, bridge, MarketEvent.BuyOrderCreated, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.BuyOrderCreated, payload)

    assert len(rec.calls) == 2
    assert rec.calls[0] == ("buy_created", payload)
    assert rec.calls[1] == ("buy_created", payload)


def test_sell_order_created_parity(pubsub_pair, event_recorder) -> None:
    """SellOrderCreated fires to one listener on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("sell_created")
    register_on_both(legacy, bridge, MarketEvent.SellOrderCreated, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.SellOrderCreated, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


# ---------------------------------------------------------------------------
# Order-completion events
# ---------------------------------------------------------------------------


def test_buy_order_completed_parity(pubsub_pair, event_recorder) -> None:
    """BuyOrderCompleted dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("buy_completed")
    register_on_both(legacy, bridge, MarketEvent.BuyOrderCompleted, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.BuyOrderCompleted, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


def test_sell_order_completed_parity(pubsub_pair, event_recorder) -> None:
    """SellOrderCompleted dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("sell_completed")
    register_on_both(legacy, bridge, MarketEvent.SellOrderCompleted, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.SellOrderCompleted, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


# ---------------------------------------------------------------------------
# Order lifecycle events
# ---------------------------------------------------------------------------


def test_order_cancelled_parity(pubsub_pair, event_recorder) -> None:
    """OrderCancelled dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("cancelled")
    register_on_both(legacy, bridge, MarketEvent.OrderCancelled, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.OrderCancelled, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


def test_order_filled_parity(pubsub_pair, event_recorder) -> None:
    """OrderFilled dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("filled")
    register_on_both(legacy, bridge, MarketEvent.OrderFilled, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.OrderFilled, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


def test_order_expired_parity(pubsub_pair, event_recorder) -> None:
    """OrderExpired dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("expired")
    register_on_both(legacy, bridge, MarketEvent.OrderExpired, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.OrderExpired, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


def test_order_failure_parity(pubsub_pair, event_recorder) -> None:
    """OrderFailure dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("order_fail")
    register_on_both(legacy, bridge, MarketEvent.OrderFailure, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.OrderFailure, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


# ---------------------------------------------------------------------------
# Asset / transaction events
# ---------------------------------------------------------------------------


def test_received_asset_parity(pubsub_pair, event_recorder) -> None:
    """ReceivedAsset dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("recv_asset")
    register_on_both(legacy, bridge, MarketEvent.ReceivedAsset, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.ReceivedAsset, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


def test_withdraw_asset_parity(pubsub_pair, event_recorder) -> None:
    """WithdrawAsset dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("withdraw")
    register_on_both(legacy, bridge, MarketEvent.WithdrawAsset, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.WithdrawAsset, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


def test_transaction_failure_parity(pubsub_pair, event_recorder) -> None:
    """TransactionFailure dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("tx_fail")
    register_on_both(legacy, bridge, MarketEvent.TransactionFailure, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.TransactionFailure, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


# ---------------------------------------------------------------------------
# Funding event
# ---------------------------------------------------------------------------


def test_funding_payment_completed_parity(pubsub_pair, event_recorder) -> None:
    """FundingPaymentCompleted dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("funding")
    register_on_both(legacy, bridge, MarketEvent.FundingPaymentCompleted, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.FundingPaymentCompleted, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


# ---------------------------------------------------------------------------
# Range-position events
# ---------------------------------------------------------------------------


def test_range_position_liquidity_added_parity(pubsub_pair, event_recorder) -> None:
    """RangePositionLiquidityAdded dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("range_add")
    register_on_both(legacy, bridge, MarketEvent.RangePositionLiquidityAdded, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.RangePositionLiquidityAdded, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


def test_range_position_liquidity_removed_parity(pubsub_pair, event_recorder) -> None:
    """RangePositionLiquidityRemoved dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("range_rem")
    register_on_both(legacy, bridge, MarketEvent.RangePositionLiquidityRemoved, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.RangePositionLiquidityRemoved, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


def test_range_position_update_failure_parity(pubsub_pair, event_recorder) -> None:
    """RangePositionUpdateFailure dispatches identically on both buses."""
    legacy, bridge = pubsub_pair
    rec = event_recorder("range_fail")
    register_on_both(legacy, bridge, MarketEvent.RangePositionUpdateFailure, rec)

    payload = object()
    _fire_both(legacy, bridge, MarketEvent.RangePositionUpdateFailure, payload)

    assert len(rec.calls) == 2
    assert all(p == payload for _, p in rec.calls)


# ---------------------------------------------------------------------------
# Cross-tag isolation — firing one MARKET_EVENTS tag must not bleed into
# listeners registered on a different tag (bridge-specific risk because
# topic strings must be distinct per tag).
# ---------------------------------------------------------------------------


def test_cross_tag_isolation_created_vs_cancelled(pubsub_pair, event_recorder) -> None:
    """Listeners on BuyOrderCreated must not receive OrderCancelled events."""
    legacy, bridge = pubsub_pair
    created_rec = event_recorder("created")
    cancelled_rec = event_recorder("cancelled")

    register_on_both(legacy, bridge, MarketEvent.BuyOrderCreated, created_rec)
    register_on_both(legacy, bridge, MarketEvent.OrderCancelled, cancelled_rec)

    payload_c = object()
    payload_x = object()
    _fire_both(legacy, bridge, MarketEvent.BuyOrderCreated, payload_c)
    _fire_both(legacy, bridge, MarketEvent.OrderCancelled, payload_x)

    # created_rec sees only BuyOrderCreated
    assert all(p == payload_c for _, p in created_rec.calls)
    assert len(created_rec.calls) == 2

    # cancelled_rec sees only OrderCancelled
    assert all(p == payload_x for _, p in cancelled_rec.calls)
    assert len(cancelled_rec.calls) == 2


def test_all_market_events_tags_fire(pubsub_pair, event_recorder) -> None:
    """All 15 ConnectorBase.MARKET_EVENTS tags dispatch via both buses."""
    from hummingbot.connector.connector_base import ConnectorBase

    legacy, bridge = pubsub_pair
    fired: dict[MarketEvent, list[object]] = {}

    for tag in ConnectorBase.MARKET_EVENTS:
        rec = event_recorder(tag.name)
        register_on_both(legacy, bridge, tag, rec)
        payload = object()
        _fire_both(legacy, bridge, tag, payload)
        fired[tag] = (rec, payload)

    for tag, (rec, payload) in fired.items():
        assert len(rec.calls) == 2, f"{tag.name}: expected 2 calls, got {len(rec.calls)}"
        assert all(p == payload for _, p in rec.calls), f"{tag.name}: payload mismatch"
