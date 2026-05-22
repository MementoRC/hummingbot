"""Data type for delayed/conditional market orders.

Delayed market orders (stop-loss, take-profit, trailing-stop) don't have
exchange order IDs until they trigger. This class tracks the pending state.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from hummingbot.core.data_type.common import OrderType, TradeType


class DelayedOrderStatus(Enum):
    PENDING = "pending"  # Placed on exchange, waiting for trigger
    TRIGGERED = "triggered"  # Price condition met, executing
    FILLED = "filled"  # Execution complete
    CANCELLED = "cancelled"  # Cancelled before trigger
    FAILED = "failed"  # Failed to place or execute


@dataclass
class DelayedMarketOrder:
    """Represents a conditional order placed on the exchange.

    Unlike regular orders, delayed orders don't have an exchange_order_id
    until they trigger. The client_order_id is assigned locally.
    """

    client_order_id: str
    trading_pair: str
    order_type: OrderType
    trade_type: TradeType
    amount: Decimal
    trigger_price: Decimal
    status: DelayedOrderStatus = DelayedOrderStatus.PENDING
    exchange_order_id: str | None = None  # Only set after trigger

    @property
    def is_pending(self) -> bool:
        return self.status == DelayedOrderStatus.PENDING

    @property
    def is_done(self) -> bool:
        return self.status in (
            DelayedOrderStatus.FILLED,
            DelayedOrderStatus.CANCELLED,
            DelayedOrderStatus.FAILED,
        )
