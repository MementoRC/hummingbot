"""Translate market-connector core events (plain dataclasses, published on an EventBus under integer ids) into
the parent hummingbot event classes dispatched through ``trigger_event``."""

from collections import OrderedDict
import dataclasses
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.core.data_type.trade_fee import TradeFeeBase, TradeFeeSchema
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

# Bus keys are the integer MarketEvent values (market_connector.hb_compat.order_events.MARKET_EVENT_IDS).
EVENT_CLASSES: Dict[int, Tuple[MarketEvent, type]] = {
    102: (MarketEvent.BuyOrderCompleted, BuyOrderCompletedEvent),
    103: (MarketEvent.SellOrderCompleted, SellOrderCompletedEvent),
    106: (MarketEvent.OrderCancelled, OrderCancelledEvent),
    107: (MarketEvent.OrderFilled, OrderFilledEvent),
    198: (MarketEvent.OrderFailure, MarketOrderFailureEvent),
    200: (MarketEvent.BuyOrderCreated, BuyOrderCreatedEvent),
    201: (MarketEvent.SellOrderCreated, SellOrderCreatedEvent),
}
MARKET_EVENT_BY_ID: Dict[int, MarketEvent] = {event_id: pair[0] for event_id, pair in EVENT_CLASSES.items()}
TERMINAL_EVENT_IDS = frozenset({102, 103, 106, 198})

_ENUM_FIELDS = {"order_type": OrderType, "type": OrderType, "trade_type": TradeType}


def to_parent_order_type(value: Any) -> OrderType:
    return OrderType[value.name]


def to_parent_trade_type(value: Any) -> TradeType:
    return TradeType[value.name]


def to_parent_position_action(value: Any) -> PositionAction:
    return PositionAction[value.name]


def to_parent_fee(fee: Any, trade_type: Any) -> TradeFeeBase:
    """Convert a primitive trade fee to the parent class (JSON round trip); fall back to a bare spot fee."""
    if isinstance(fee, TradeFeeBase):
        return fee
    try:
        return TradeFeeBase.from_json(fee.to_json())
    except (AttributeError, KeyError, TypeError, ValueError, ArithmeticError):
        percent = getattr(fee, "percent", Decimal(0))
        return TradeFeeBase.new_spot_fee(
            fee_schema=TradeFeeSchema(), trade_type=to_parent_trade_type(trade_type), percent=percent
        )


def _field_names(cls: type) -> Tuple[str, ...]:
    fields = getattr(cls, "_fields", None)
    return tuple(fields) if fields is not None else tuple(f.name for f in dataclasses.fields(cls))


def to_parent_event(event_id: int, mc_event: Any) -> Any:
    """Build the parent event for bus key ``event_id`` from the core's event; unknown ids raise ``KeyError``."""
    parent_cls = EVENT_CLASSES[event_id][1]
    kwargs: Dict[str, Any] = {}
    for name in _field_names(parent_cls):
        value = getattr(mc_event, name)
        if name in _ENUM_FIELDS:
            value = _ENUM_FIELDS[name][value.name]
        elif name == "trade_fee":
            value = to_parent_fee(value, mc_event.trade_type)
        kwargs[name] = value
    return parent_cls(**kwargs)


class TerminalEventDeduper:
    """Bounded memory of (event_id, client_order_id) terminal events already forwarded."""

    def __init__(self, max_entries: int = 10_000):
        self._max = max_entries
        self._seen: "OrderedDict[Tuple[int, str], None]" = OrderedDict()

    def is_duplicate(self, event_id: int, client_order_id: Optional[str]) -> bool:
        """True when this terminal event was seen before; otherwise remembers it. Non-terminal ids never match."""
        if event_id not in TERMINAL_EVENT_IDS:
            return False
        key = (event_id, str(client_order_id))
        if key in self._seen:
            return True
        self._seen[key] = None
        while len(self._seen) > self._max:
            self._seen.popitem(last=False)
        return False

    def clear(self) -> None:
        self._seen.clear()
