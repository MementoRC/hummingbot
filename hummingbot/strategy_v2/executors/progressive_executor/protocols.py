from decimal import Decimal
from typing import Protocol, List, Union

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import TradeType
from hummingbot.core.event.events import BuyOrderCreatedEvent, SellOrderCreatedEvent, BuyOrderCompletedEvent, \
    SellOrderCompletedEvent, OrderFilledEvent, OrderCancelledEvent, MarketOrderFailureEvent
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.strategy_v2.executors.executor_protocols import ExecutorProtocol
from hummingbot.strategy_v2.executors.progressive_executor.data_types import ProgressiveExecutorConfig
from hummingbot.strategy_v2.models.executors import TrackedOrder, CloseType


class ProgressiveProtocol(Protocol):
    """
    This protocol defines the methods that are required to be implemented by the Progressive Executor.

    :param config: The configuration of the Progressive Executor.
    :param total_executed_amount_backup: The total executed amount of the open order.
    :param current_retries: The number of retries that have been attempted.
    :param open_order: The open order that is being tracked.
    :param realized_orders: The list of realized orders.
    :param close_order: The close order that is being tracked.
    :param failed_orders: The list of failed orders.
    :param canceled_orders: The list of canceled orders.
    :param open_order_timestamp: The timestamp of the open order.
    :param trailing_stop_trigger_pnl: The trailing stop trigger percentage.
    """
    config: ProgressiveExecutorConfig
    total_executed_amount_backup: Decimal
    current_retries: int
    open_order: TrackedOrder | None
    realized_orders: List[TrackedOrder]
    close_order: TrackedOrder | None
    failed_orders: List[TrackedOrder]
    canceled_orders: List[TrackedOrder]
    open_order_timestamp: float | int | None
    trailing_stop_trigger_pnl: Decimal | None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        ...

    @property
    def strategy(self) -> ScriptStrategyBase:
        ...

    @property
    def is_expired(self) -> bool:
        ...

    @property
    def is_extended_on_yield(self) -> bool:
        ...

    @property
    def entry_price(self) -> Decimal:
        ...

    @property
    def close_price(self) -> Decimal:
        ...

    @property
    def side(self) -> TradeType:
        ...

    @property
    def current_timestamp(self) -> float:
        ...

    @property
    def max_retries(self) -> int:
        ...

    def _is_within_activation_bounds(self, close_price: Decimal) -> bool:
        ...


class PNLCalculatorProtocol(Protocol):
    @property
    def trade_pnl_pct(self) -> Decimal:
        ...

    @property
    def trade_pnl_quote(self) -> Decimal:
        ...

    def get_net_pnl_quote(self) -> Decimal:
        ...

    def get_cum_fees_quote(self) -> Decimal:
        ...

    def get_net_pnl_pct(self) -> Decimal:
        ...

    def get_target_pnl_yield(self) -> Decimal:
        ...


class OrderManagementProtocol(Protocol):
    @property
    def open_filled_amount(self) -> Decimal:
        ...

    @property
    def open_filled_amount_quote(self) -> Decimal:
        ...

    def open_orders_completed(self) -> bool:
        ...

    @property
    def realized_filled_amount(self) -> Decimal:
        ...

    @property
    def unrealized_filled_amount(self) -> Decimal:
        ...

    @property
    def close_filled_amount(self) -> Decimal:
        ...

    @property
    def close_filled_amount_quote(self) -> Decimal:
        ...

    @property
    def filled_amount(self) -> Decimal:
        ...

    @property
    def filled_amount_quote(self) -> Decimal:
        ...

    def update_tracked_orders_with_order_id(self, order_id: str) -> None:
        ...


class ProgressiveOrderPNLProtocol(
    ExecutorProtocol,
    ProgressiveProtocol,
    OrderManagementProtocol,
    PNLCalculatorProtocol
):
    pass


class OrderExecutionProtocol(Protocol):
    def place_open_order(self) -> None:
        ...

    def place_close_order_and_cancel_open_orders(
            self,
            close_type: CloseType,
            price: Decimal = Decimal("NaN")
    ):
        ...

    def place_partial_close_order(
            self,
            close_type: CloseType,
            price: Decimal = Decimal("NaN"),
            amount_to_close: Decimal = Decimal("NaN")
    ) -> None:
        ...

    def cancel_open_order(self) -> None:
        ...

    def cancel_close_order(self) -> None:
        ...

    def cancel_open_orders(self) -> None:
        ...


class ProgressiveOrderProtocol(
    OrderManagementProtocol,
    ProgressiveProtocol,
    ExecutorProtocol,
):
    pass


class ProgressiveOrderExecutionProtocol(
    OrderManagementProtocol,
    OrderExecutionProtocol,
    ProgressiveProtocol,
    ExecutorProtocol,
):
    pass


class OrderProcessingProtocol(Protocol):
    current_retry: int

    def process_order_created_event(self, _, market, event: Union[BuyOrderCreatedEvent, SellOrderCreatedEvent]):
        ...

    def process_order_completed_event(self, _, market, event: Union[BuyOrderCompletedEvent, SellOrderCompletedEvent]):
        ...

    def process_order_filled_event(self, _, market, event: OrderFilledEvent):
        ...

    def process_order_canceled_event(self, _, market: ConnectorBase, event: OrderCancelledEvent):
        ...

    def process_order_failed_event(self, _, market, event: MarketOrderFailureEvent):
        ...


class ProgressiveOrderProcessProtocol(
    OrderManagementProtocol,
    OrderProcessingProtocol,
    ProgressiveProtocol,
    ExecutorProtocol,
):
    pass


class ControlProtocol(Protocol):
    def control_task(self) -> None:
        ...

    def control_barriers(self) -> None:
        ...

    def control_stop_loss(self) -> None:
        ...

    def control_trailing_stop(self) -> None:
        ...

    def control_time_limit(self) -> None:
        ...

    def control_open_order(self) -> None:
        ...

    def control_failed_orders(self) -> None:
        ...

    async def control_shutdown_process(self) -> None:
        ...

    def evaluate_max_retries(self) -> None:
        ...


class ProgressiveControlProtocol(ProgressiveProtocol, ControlProtocol, ExecutorProtocol):
    pass


class ProgressiveOrderControlProtocol(
    ProgressiveProtocol,
    ControlProtocol,
    OrderExecutionProtocol,
    OrderManagementProtocol,
    ExecutorProtocol,
):
    pass


class ProgressiveOrderExecutionPNLControlProtocol(
    ProgressiveProtocol,
    OrderManagementProtocol,
    OrderExecutionProtocol,
    PNLCalculatorProtocol,
    ControlProtocol,
    ExecutorProtocol,
):
    pass
