from decimal import Decimal

from hummingbot.core.data_type.common import PositionAction, OrderType, TradeType
from hummingbot.strategy_v2.executors.progressive_executor.protocols import ProgressiveOrderProtocol, \
    ProgressiveOrderExecutionProtocol
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import TrackedOrder, CloseType


class OrderExecutionMixin:
    def place_open_order(self: ProgressiveOrderProtocol) -> None:
        """
        This method is responsible for placing the open order.

        :return: None
        """
        self.logger().debug("Attempting to place open order...")
        order_id = self.place_order(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            order_type=self.config.triple_barrier_config.open_order_type,
            amount=self.config.amount,
            price=self.entry_price,
            side=self.config.side,
            position_action=PositionAction.OPEN,
        )
        self.logger().debug(f"Open order placed successfully - Order ID: {order_id}")
        self.open_order = TrackedOrder(order_id=order_id)
        self.open_order_timestamp = self.current_timestamp
        self.logger().debug("Placing open order")

    def place_trailing_stop_order(self: ProgressiveOrderProtocol) -> None:
        """
        This method is responsible for placing the open order.

        :return: None
        """
        order_id = self.place_order(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            order_type=self.config.triple_barrier_config.trailing_stop_order_type,
            amount=self.config.amount,
            price=Decimal("NaN"),
            side=self.config.side,
            position_action=PositionAction.OPEN,
            price_pct_offset=self.config.triple_barrier_config.stop_loss
        )
        self.open_order = TrackedOrder(order_id=order_id)
        self.open_order_timestamp = self.current_timestamp
        self.logger().debug("Placing trailing-stop order")

    def place_close_order_and_cancel_open_orders(
            self: ProgressiveOrderExecutionProtocol,
            close_type: CloseType,
            price: Decimal = Decimal("NaN")
    ):
        """
        This method is responsible for placing the close order and canceling the open orders. If the difference between
        the open filled amount and the close filled amount is greater than the minimum order size, it places the close
        order. It also cancels the open orders.

        :param close_type: The type of the close order.
        :param price: The price to be used in the close order.
        :return: None
        """
        delta_amount_to_close = self.unrealized_filled_amount - self.close_filled_amount
        trading_rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        self.cancel_open_orders()
        if delta_amount_to_close > trading_rules.min_order_size:
            try:
                order_id = self.place_order(
                    connector_name=self.config.connector_name,
                    trading_pair=self.config.trading_pair,
                    order_type=OrderType.MARKET,
                    amount=delta_amount_to_close,
                    price=price,
                    side=TradeType.SELL if self.config.side == TradeType.BUY else TradeType.BUY,
                    position_action=PositionAction.CLOSE,
                )
                self.close_order = TrackedOrder(order_id=order_id)
                self.logger().debug(f"Placing close order --> Filled amount: {self.open_filled_amount}")
            except Exception as e:
                self.logger().error(f"Failed to place close order: {e}")

        self.close_type = close_type
        self.close_timestamp = self.current_timestamp
        self._status = RunnableStatus.SHUTTING_DOWN

    def place_partial_close_order(
            self: ProgressiveOrderExecutionProtocol,
            close_type: CloseType,
            price: Decimal = Decimal("NaN"),
            amount_to_close: Decimal = Decimal("NaN")
    ) -> None:
        """
        This method is responsible for placing a partial close order.
         If the amount is larger than the difference between
        the open filled amount and the close filled amount, it places the close/cancel.

        :param close_type: The type of the close order.
        :param price: The price to be used in the close order.
        :param amount_to_close: The amount to be closed.
        :return: None
        """
        if amount_to_close >= self.unrealized_filled_amount:
            self.place_close_order_and_cancel_open_orders(close_type=close_type, price=price)
            return

        trading_rules = self.get_trading_rules(self.config.connector_name, self.config.trading_pair)
        if amount_to_close > trading_rules.min_order_size:
            try:
                order_id = self.place_order(
                    connector_name=self.config.connector_name,
                    trading_pair=self.config.trading_pair,
                    order_type=OrderType.MARKET,
                    amount=amount_to_close,
                    price=price,
                    side=TradeType.SELL if self.config.side == TradeType.BUY else TradeType.BUY,
                    position_action=PositionAction.NIL,
                )
                self.realized_orders.append(TrackedOrder(order_id=order_id))
                self.logger().debug(f"Placing partial close order --> Filled amount: {amount_to_close}")
            except Exception as e:
                self.logger().error(f"Failed to place partial close order: {e}")

    def cancel_open_order(self: ProgressiveOrderProtocol) -> None:
        """
        This method is responsible for canceling the open order.

        :return: None
        """
        self.strategy.cancel(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            order_id=self.open_order.order_id
        )
        self.logger().debug("Removing open order")

    def cancel_close_order(self: ProgressiveOrderProtocol) -> None:
        """
        This method is responsible for canceling the close order.

        :return: None
        """
        self.strategy.cancel(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            order_id=self.close_order.order_id
        )
        self.logger().debug("Removing close order")

    def cancel_open_orders(self: ProgressiveOrderExecutionProtocol) -> None:
        """
        This method is responsible for canceling the open orders.

        :return: None
        """
        if self.open_order and self.open_order.order and self.open_order.order.is_open:
            self.cancel_open_order()
