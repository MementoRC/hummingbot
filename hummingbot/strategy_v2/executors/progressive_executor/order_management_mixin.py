from decimal import Decimal

from hummingbot.strategy_v2.executors.progressive_executor.protocols import OrderManagementProtocol, \
    ProgressiveOrderProtocol


class OrderManagementMixin:
    @property
    def open_filled_amount(self: ProgressiveOrderProtocol) -> Decimal:
        """
        Get the filled amount of the open order.

        :return: The filled amount of the open order if it exists, otherwise 0.
        """
        return self.open_order.executed_amount_base if self.open_order else Decimal("0")

    @property
    def open_filled_amount_quote(self: ProgressiveOrderProtocol) -> Decimal:
        """
        Get the filled amount of the open order in quote asset.

        :return: The filled amount of the open order in quote asset if it exists, otherwise 0.
        """
        return self.open_filled_amount * self.entry_price

    def open_orders_completed(self: ProgressiveOrderProtocol) -> bool:
        """
        This method is responsible for checking if the open orders are completed.

        :return: True if the open orders are completed, False otherwise.
        """
        open_order_condition = not self.open_order or self.open_order.is_done
        failed_orders_condition = not self.failed_orders or all(order.is_done for order in self.failed_orders)
        return open_order_condition and failed_orders_condition

    @property
    def realized_filled_amount(self: ProgressiveOrderProtocol) -> Decimal:
        """
        Get the realized filled amount of the intermediate close orders.

        :return: The filled amount of the close order if it exists, otherwise 0.
        """
        return sum(
            (order.executed_amount_base for order in self.realized_orders),
            start=Decimal("0"),
        )

    @property
    def unrealized_filled_amount(self) -> Decimal:
        """
        Get the realized filled amount of the intermediate close orders.

        :return: The filled amount of the close order if it exists, otherwise 0.
        """
        return self.open_filled_amount - self.realized_filled_amount

    @property
    def close_filled_amount(self: ProgressiveOrderProtocol) -> Decimal:
        """
        Get the filled amount of the close order.

        :return: The filled amount of the close order if it exists, otherwise 0.
        """
        return self.close_order.executed_amount_base if self.close_order else Decimal("0")

    @property
    def close_filled_amount_quote(self: ProgressiveOrderProtocol) -> Decimal:
        """
        Get the filled amount of the close order in quote currency.

        :return: The filled amount of the close order in quote currency.
        """
        return self.close_filled_amount * self.close_price

    @property
    def filled_amount(self) -> Decimal:
        """
        Get the filled amount of the position.
        """
        return self.open_filled_amount + self.close_filled_amount + self.realized_filled_amount

    @property
    def filled_amount_quote(self) -> Decimal:
        """
        Get the filled amount of the position in quote currency.
        """
        return self.open_filled_amount_quote + self.close_filled_amount_quote

    def update_tracked_orders_with_order_id(self: ProgressiveOrderProtocol, order_id: str):
        """
        This method is responsible for updating the tracked orders with the information from the InFlightOrder, using
        the order_id as a reference.

        :param order_id: The order_id to be used as a reference.
        :return: None
        """
        if self.open_order and self.open_order.order_id == order_id:
            self.open_order.order = self.get_in_flight_order(self.config.connector_name, order_id)
        elif self.close_order and self.close_order.order_id == order_id:
            self.close_order.order = self.get_in_flight_order(self.config.connector_name, order_id)
        else:
            for order in self.realized_orders:
                if order.order_id == order_id:
                    order.order = self.get_in_flight_order(self.config.connector_name, order_id)
