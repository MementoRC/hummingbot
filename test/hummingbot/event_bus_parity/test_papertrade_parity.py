"""B13 — PaperTradeExchange event dispatch parity tests.

Verifies that PaperTradeExchange (legacy PubSub path, driven via trigger_event)
and PubSubBridge.c_trigger_event (EventBus path) produce identical dispatch
semantics for each MarketEvent tag the exchange fires internally.

Note: PaperTradeExchange.c_trigger_event is a Cython cdef method — not
Python-callable from test code.  trigger_event is the Python-accessible
equivalent and produces the same dispatch semantics.  The bridge exposes
c_trigger_event as a pure-Python alias and is called directly here to
verify that alias works correctly.

Strategy: drive both sides with the same RecorderListener, then assert
``calls`` lists match.  The PaperTradeExchange instance is constructed with
mocked heavy dependencies so no live network or Cython compilation is
required at test-collection time.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from event_bus import EventBus

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.trade_fee import TokenAmount, TradeFeeBase
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
from hummingbot.core.event_bus_bridge import PubSubBridge

from .conftest import RecorderListener

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

TIMESTAMP = 1_700_000_000.0
TRADING_PAIR = "BTC-USDT"
ORDER_ID = "buy://BTC-USDT/aabbccddeeff"
BASE = "BTC"
QUOTE = "USDT"
AMOUNT = Decimal("0.5")
PRICE = Decimal("30000")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_fee() -> TradeFeeBase:
    """Return a minimal flat trade fee suitable for OrderFilledEvent."""
    from hummingbot.core.data_type.trade_fee import AddedToCostTradeFee

    return AddedToCostTradeFee(flat_fees=[TokenAmount(QUOTE, Decimal("0"))])


def _make_bridge() -> PubSubBridge:
    return PubSubBridge(EventBus(name="parity-test"))


def _make_exchange():
    """Return a PaperTradeExchange with all heavy deps mocked.

    We only need the PubSub machinery (inherited via ExchangeBase) and the
    class-level event-tag constants.  The OrderBookTracker and target_market
    are never exercised in these parity tests.
    """
    from hummingbot.connector.exchange.paper_trade.paper_trade_exchange import PaperTradeExchange

    mock_tracker = MagicMock()
    mock_tracker.data_source.order_book_create_function = None
    mock_target = MagicMock()

    with patch("hummingbot.connector.exchange.paper_trade.paper_trade_exchange.BudgetChecker"):
        exchange = PaperTradeExchange(
            order_book_tracker=mock_tracker,
            target_market=mock_target,
            exchange_name="paper_test",
        )
    return exchange


# ---------------------------------------------------------------------------
# Parametrised parity helper
# ---------------------------------------------------------------------------


def _assert_parity(
    event_tag: MarketEvent,
    event_obj: object,
) -> None:
    """Register one RecorderListener on both sides, fire events, compare."""
    exchange = _make_exchange()
    bridge = _make_bridge()

    legacy_rec = RecorderListener("legacy")
    bridge_rec = RecorderListener("bridge")

    # Register on legacy exchange (PubSub path)
    exchange.add_listener(event_tag, legacy_rec)
    # Register on bridge (EventBus path) — tag.value is the int
    bridge.add_listener(event_tag.value, bridge_rec)

    # Fire on legacy — PaperTradeExchange.c_trigger_event is Cython cdef, not
    # Python-callable; trigger_event is the Python-accessible equivalent.
    exchange.trigger_event(event_tag.value, event_obj)
    # Fire on bridge — bridge.c_trigger_event is a pure-Python alias
    bridge.c_trigger_event(event_tag.value, event_obj)

    # Both recorders must have captured exactly one call with identical payload
    assert len(legacy_rec.calls) == 1, (
        f"Legacy PubSub expected 1 call for {event_tag.name}, got {len(legacy_rec.calls)}"
    )
    assert len(bridge_rec.calls) == 1, f"PubSubBridge expected 1 call for {event_tag.name}, got {len(bridge_rec.calls)}"
    _, legacy_payload = legacy_rec.calls[0]
    _, bridge_payload = bridge_rec.calls[0]
    assert legacy_payload is bridge_payload, (
        f"Payload identity mismatch for {event_tag.name}: legacy={legacy_payload!r}, bridge={bridge_payload!r}"
    )


# ---------------------------------------------------------------------------
# Individual parity tests — one per MarketEvent tag
# ---------------------------------------------------------------------------


def test_order_filled_parity() -> None:
    """OrderFilledEvent dispatches identically on legacy and bridge."""
    event = OrderFilledEvent(
        timestamp=TIMESTAMP,
        order_id=ORDER_ID,
        trading_pair=TRADING_PAIR,
        trade_type=TradeType.BUY,
        order_type=OrderType.MARKET,
        price=PRICE,
        amount=AMOUNT,
        trade_fee=_flat_fee(),
        exchange_trade_id="eid_001",
    )
    _assert_parity(MarketEvent.OrderFilled, event)


def test_buy_order_completed_parity() -> None:
    """BuyOrderCompletedEvent dispatches identically on legacy and bridge."""
    event = BuyOrderCompletedEvent(
        timestamp=TIMESTAMP,
        order_id=ORDER_ID,
        base_asset=BASE,
        quote_asset=QUOTE,
        base_asset_amount=AMOUNT,
        quote_asset_amount=PRICE * AMOUNT,
        order_type=OrderType.MARKET,
    )
    _assert_parity(MarketEvent.BuyOrderCompleted, event)


def test_sell_order_completed_parity() -> None:
    """SellOrderCompletedEvent dispatches identically on legacy and bridge."""
    event = SellOrderCompletedEvent(
        timestamp=TIMESTAMP,
        order_id=ORDER_ID,
        base_asset=BASE,
        quote_asset=QUOTE,
        base_asset_amount=AMOUNT,
        quote_asset_amount=PRICE * AMOUNT,
        order_type=OrderType.MARKET,
    )
    _assert_parity(MarketEvent.SellOrderCompleted, event)


def test_order_cancelled_parity() -> None:
    """OrderCancelledEvent dispatches identically on legacy and bridge."""
    event = OrderCancelledEvent(
        timestamp=TIMESTAMP,
        order_id=ORDER_ID,
    )
    _assert_parity(MarketEvent.OrderCancelled, event)


def test_order_failure_parity() -> None:
    """MarketOrderFailureEvent dispatches identically on legacy and bridge."""
    event = MarketOrderFailureEvent(
        timestamp=TIMESTAMP,
        order_id=ORDER_ID,
        order_type=OrderType.MARKET,
    )
    _assert_parity(MarketEvent.OrderFailure, event)


def test_buy_order_created_parity() -> None:
    """BuyOrderCreatedEvent dispatches identically on legacy and bridge."""
    event = BuyOrderCreatedEvent(
        timestamp=TIMESTAMP,
        type=OrderType.LIMIT,
        trading_pair=TRADING_PAIR,
        amount=AMOUNT,
        price=PRICE,
        order_id=ORDER_ID,
        creation_timestamp=TIMESTAMP,
    )
    _assert_parity(MarketEvent.BuyOrderCreated, event)


def test_sell_order_created_parity() -> None:
    """SellOrderCreatedEvent dispatches identically on legacy and bridge."""
    event = SellOrderCreatedEvent(
        timestamp=TIMESTAMP,
        type=OrderType.LIMIT,
        trading_pair=TRADING_PAIR,
        amount=AMOUNT,
        price=PRICE,
        order_id=ORDER_ID,
        creation_timestamp=TIMESTAMP,
    )
    _assert_parity(MarketEvent.SellOrderCreated, event)


# ---------------------------------------------------------------------------
# Multi-listener parity — two listeners on each side
# ---------------------------------------------------------------------------


def test_multi_listener_order_filled_parity() -> None:
    """Two listeners on each side both receive OrderFilledEvent."""
    event = OrderFilledEvent(
        timestamp=TIMESTAMP,
        order_id=ORDER_ID,
        trading_pair=TRADING_PAIR,
        trade_type=TradeType.SELL,
        order_type=OrderType.LIMIT,
        price=PRICE,
        amount=AMOUNT,
        trade_fee=_flat_fee(),
        exchange_trade_id="eid_002",
    )

    exchange = _make_exchange()
    bridge = _make_bridge()

    rec_a_leg = RecorderListener("a_legacy")
    rec_b_leg = RecorderListener("b_legacy")
    rec_a_br = RecorderListener("a_bridge")
    rec_b_br = RecorderListener("b_bridge")

    tag = MarketEvent.OrderFilled
    exchange.add_listener(tag, rec_a_leg)
    exchange.add_listener(tag, rec_b_leg)
    bridge.add_listener(tag.value, rec_a_br)
    bridge.add_listener(tag.value, rec_b_br)

    exchange.trigger_event(tag.value, event)
    bridge.c_trigger_event(tag.value, event)

    assert len(rec_a_leg.calls) == 1
    assert len(rec_b_leg.calls) == 1
    assert len(rec_a_br.calls) == 1
    assert len(rec_b_br.calls) == 1
    # All four calls carried the same event object
    for rec in (rec_a_leg, rec_b_leg, rec_a_br, rec_b_br):
        assert rec.calls[0][1] is event


# ---------------------------------------------------------------------------
# Unsubscribe parity — removed listener must not receive further events
# ---------------------------------------------------------------------------


def test_remove_listener_parity() -> None:
    """remove_listener stops delivery on both legacy and bridge."""
    event = OrderCancelledEvent(timestamp=TIMESTAMP, order_id=ORDER_ID)

    exchange = _make_exchange()
    bridge = _make_bridge()

    rec_leg = RecorderListener("leg")
    rec_br = RecorderListener("br")

    tag = MarketEvent.OrderCancelled
    exchange.add_listener(tag, rec_leg)
    bridge.add_listener(tag.value, rec_br)

    # Fire once — both receive
    exchange.trigger_event(tag.value, event)
    bridge.c_trigger_event(tag.value, event)
    assert len(rec_leg.calls) == 1
    assert len(rec_br.calls) == 1

    # Remove listeners
    exchange.remove_listener(tag, rec_leg)
    bridge.remove_listener(tag.value, rec_br)

    # Fire again — neither receives
    exchange.trigger_event(tag.value, event)
    bridge.c_trigger_event(tag.value, event)
    assert len(rec_leg.calls) == 1, "Legacy listener received after remove_listener"
    assert len(rec_br.calls) == 1, "Bridge listener received after remove_listener"


# ---------------------------------------------------------------------------
# Cross-tag isolation parity
# ---------------------------------------------------------------------------


def test_cross_tag_isolation_parity() -> None:
    """Listener on OrderFilled must not fire when OrderCancelled is emitted."""
    filled_event = OrderFilledEvent(
        timestamp=TIMESTAMP,
        order_id=ORDER_ID,
        trading_pair=TRADING_PAIR,
        trade_type=TradeType.BUY,
        order_type=OrderType.MARKET,
        price=PRICE,
        amount=AMOUNT,
        trade_fee=_flat_fee(),
        exchange_trade_id="eid_003",
    )
    cancelled_event = OrderCancelledEvent(timestamp=TIMESTAMP, order_id=ORDER_ID)

    exchange = _make_exchange()
    bridge = _make_bridge()

    rec_leg = RecorderListener("leg_filled")
    rec_br = RecorderListener("br_filled")

    # Only subscribe to OrderFilled
    exchange.add_listener(MarketEvent.OrderFilled, rec_leg)
    bridge.add_listener(MarketEvent.OrderFilled.value, rec_br)

    # Fire OrderCancelled — must not reach OrderFilled listeners
    exchange.trigger_event(MarketEvent.OrderCancelled.value, cancelled_event)
    bridge.c_trigger_event(MarketEvent.OrderCancelled.value, cancelled_event)

    assert len(rec_leg.calls) == 0, "Legacy: cross-tag leak on OrderFilled listener"
    assert len(rec_br.calls) == 0, "Bridge: cross-tag leak on OrderFilled listener"

    # Confirm OrderFilled still fires correctly
    exchange.trigger_event(MarketEvent.OrderFilled.value, filled_event)
    bridge.c_trigger_event(MarketEvent.OrderFilled.value, filled_event)

    assert len(rec_leg.calls) == 1
    assert len(rec_br.calls) == 1
