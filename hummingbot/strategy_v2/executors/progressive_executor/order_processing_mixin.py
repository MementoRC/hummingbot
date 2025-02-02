from typing import Union

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import BuyOrderCreatedEvent, SellOrderCreatedEvent, BuyOrderCompletedEvent, \
    SellOrderCompletedEvent, OrderFilledEvent, OrderCancelledEvent, MarketOrderFailureEvent
from hummingbot.strategy_v2.executors.progressive_executor.protocols import OrderManagementProtocol, \
    ProgressiveOrderProtocol, ProgressiveOrderProcessProtocol


class OrderProcessingMixin:
    def process_order_created_event(
            self: OrderManagementProtocol,
            _,
            market,
            event: Union[BuyOrderCreatedEvent, SellOrderCreatedEvent]
    ):
        """
        This method is responsible for processing the order created event. Here we will update the TrackedOrder with the
        order_id.
        """
        self.update_tracked_orders_with_order_id(event.order_id)

    def process_order_completed_event(self: ProgressiveOrderProtocol, _, market,
                                      event: Union[BuyOrderCompletedEvent, SellOrderCompletedEvent]):
        """
        This method is responsible for processing the order completed event. Here we will check if the id is one of the
        tracked orders and update the state
        """
        if self.close_order and self.close_order.order_id == event.order_id:
            self.close_timestamp = event.timestamp

    def process_order_filled_event(self: ProgressiveOrderProtocol, _, market, event: OrderFilledEvent):
        """
        This method is responsible for processing the order filled event. Here we will update the value of
        _total_executed_amount_backup, that can be used if the InFlightOrder
        is not available.
        """
        matching_order = OrderCandidate(
            trading_pair=self.config.trading_pair,
            is_maker=self.config.triple_barrier_config.open_order_type.is_limit_type(),
            order_type=self.config.triple_barrier_config.open_order_type,
            order_side=TradeType.SELL if self.config.side == TradeType.BUY else TradeType.BUY,
            amount=event.amount,
            price=self.entry_price,
        )
        if self.open_order and event.order_id == self.open_order.order_id:
            self.total_executed_amount_backup += event.amount
            self.lock_order_candidate(self.config.connector_name, matching_order)
        elif event.order_id in [order.order_id for order in self.realized_orders]:
            self.total_executed_amount_backup -= event.amount
            self.unlock_order_candidate(self.config.connector_name, matching_order)
        elif self.close_order and event.order_id == self.close_order.order_id:
            self.total_executed_amount_backup -= event.amount
            self.unlock_order_candidate(self.config.connector_name, matching_order)
        self.update_tracked_orders_with_order_id(event.order_id)

    def process_order_canceled_event(
            self: ProgressiveOrderProtocol,
            _,
            market: ConnectorBase,
            event: OrderCancelledEvent,
    ):
        """
        This method is responsible for processing the order canceled event
        """
        if self.close_order and event.order_id == self.close_order.order_id:
            self.canceled_orders.append(self.close_order)
            self.close_order = None
        elif any(event.order_id == order.order_id for order in self.realized_orders):
            self.canceled_orders.append(
                next(order for order in self.realized_orders if order.order_id == event.order_id))
            self.realized_orders = [order for order in self.realized_orders if order.order_id != event.order_id]

    def process_order_failed_event(
            self: ProgressiveOrderProcessProtocol,
            _,
            market: ConnectorBase,
            event: MarketOrderFailureEvent,
    ):
        """
        This method is responsible for processing the order failed event. Here we will add the InFlightOrder to the
        failed orders list.
        """
        self.current_retries += 1
        if self.open_order and event.order_id == self.open_order.order_id:
            self.failed_orders.append(self.open_order)
            self.open_order = None
            self.logger().error(f"Open order failed. Retrying {self.current_retries}/{self.max_retries}")
        elif self.close_order and event.order_id == self.close_order.order_id:
            self.failed_orders.append(self.close_order)
            self.close_order = None
            self.logger().error(f"Close order failed. Retrying {self.current_retries}/{self.max_retries}")
        elif any(event.order_id == order.order_id for order in self.realized_orders):
            self.failed_orders.append(
                next(order for order in self.realized_orders if order.order_id == event.order_id))
            self.realized_orders = [order for order in self.realized_orders if order.order_id != event.order_id]
