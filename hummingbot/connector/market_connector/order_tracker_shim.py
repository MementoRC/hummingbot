"""Read-only view of a market-connector core's orders, shaped like ``ClientOrderTracker`` for executors."""

from typing import Any, Callable, Dict, Optional, Tuple

from hummingbot.connector.market_connector.events_converter import (
    to_parent_fee,
    to_parent_order_type,
    to_parent_position_action,
    to_parent_trade_type,
)
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState, TradeUpdate

_CACHE_MAX = 2048


def to_parent_state(state: Any) -> OrderState:
    """Map by name; the primitive-only UNKNOWN is shown as PENDING_CREATE (never a state the parent persists)."""
    if state.name == "UNKNOWN":
        return OrderState.PENDING_CREATE
    return OrderState[state.name]


def to_parent_trade_update(update: Any, trade_type: Any) -> TradeUpdate:
    return TradeUpdate(
        trade_id=update.trade_id,
        client_order_id=update.client_order_id,
        exchange_order_id=update.exchange_order_id,
        trading_pair=update.trading_pair,
        fill_timestamp=update.fill_timestamp,
        fill_price=update.fill_price,
        fill_base_amount=update.fill_base_amount,
        fill_quote_amount=update.fill_quote_amount,
        fee=to_parent_fee(update.fee, trade_type),
        is_taker=update.is_taker,
    )


def to_parent_in_flight_order(order: Any) -> InFlightOrder:
    """Convert a primitive InFlightOrder; the done-ness comes from the converted core state, never recomputed."""
    converted = InFlightOrder(
        client_order_id=order.client_order_id,
        trading_pair=order.trading_pair,
        order_type=to_parent_order_type(order.order_type),
        trade_type=to_parent_trade_type(order.trade_type),
        amount=order.amount,
        creation_timestamp=order.creation_timestamp,
        price=order.price,
        exchange_order_id=order.exchange_order_id,
        initial_state=to_parent_state(order.current_state),
        leverage=order.leverage,
        position=to_parent_position_action(order.position),
    )
    converted.executed_amount_base = order.executed_amount_base
    converted.executed_amount_quote = order.executed_amount_quote
    converted.last_update_timestamp = order.last_update_timestamp
    for trade_id, update in order.order_fills.items():
        converted.order_fills[trade_id] = to_parent_trade_update(update, order.trade_type)
    converted.check_filled_condition()
    return converted


class CoreOrderTracker:
    """``core_provider`` returns the live core (or None before start); conversions are cached until the order changes."""

    def __init__(self, core_provider: Callable[[], Any]):
        self._core_provider = core_provider
        self._cache: Dict[str, Tuple[Any, tuple, InFlightOrder]] = {}

    def _convert(self, order: Any) -> InFlightOrder:
        signature = (
            order.current_state,
            order.executed_amount_base,
            order.exchange_order_id,
            order.last_update_timestamp,
            len(order.order_fills),
        )
        cached = self._cache.get(order.client_order_id)
        if cached is not None and cached[0] is order and cached[1] == signature:
            return cached[2]
        converted = to_parent_in_flight_order(order)
        self._cache.pop(order.client_order_id, None)
        self._cache[order.client_order_id] = (order, signature, converted)
        while len(self._cache) > _CACHE_MAX:
            self._cache.pop(next(iter(self._cache)))
        return converted

    @property
    def active_orders(self) -> Dict[str, InFlightOrder]:
        core = self._core_provider()
        if core is None:
            return {}
        return {cid: self._convert(order) for cid, order in list(core.in_flight_orders.items())}

    @property
    def all_fillable_orders(self) -> Dict[str, InFlightOrder]:
        return self.active_orders

    @property
    def all_updatable_orders(self) -> Dict[str, InFlightOrder]:
        return self.active_orders

    def fetch_tracked_order(self, client_order_id: str) -> Optional[InFlightOrder]:
        core = self._core_provider()
        order = None if core is None else core.in_flight_orders.get(client_order_id)
        return None if order is None else self._convert(order)

    def fetch_order(
        self, client_order_id: Optional[str] = None, exchange_order_id: Optional[str] = None
    ) -> Optional[InFlightOrder]:
        """In flight or finished (the core keeps a bounded history of finished orders)."""
        core = self._core_provider()
        if core is None:
            return None
        order = core.fetch_order(client_order_id) if client_order_id else None
        if order is None and exchange_order_id is not None:
            cid = core.client_order_id_for_txid(exchange_order_id)
            order = core.fetch_order(cid) if cid else None
        return None if order is None else self._convert(order)
