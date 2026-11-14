import asyncio
from decimal import Decimal
import logging

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    SellOrderCompletedEvent,
)
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.executor_factory import ExecutorFactory
from hummingbot.strategy_v2.executors.position_executor.position_executor import PositionExecutor
from hummingbot.strategy_v2.executors.position_on_exchange_executor.data_types import PositionOnExchangeExecutorConfig
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder


@ExecutorFactory.register(PositionOnExchangeExecutorConfig)
class PositionOnExchangeExecutor(PositionExecutor):
    """
    Executor that places stop-loss and take-profit orders directly on the exchange
    instead of monitoring barriers client-side.

    Benefits:
    - Sub-millisecond trigger (exchange-native vs 1s polling interval)
    - Survives bot disconnection (orders live on exchange)
    - Accurate execution at trigger price (no slippage from detection delay)
    """

    _logger = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(
        self,
        strategy: StrategyV2Base,
        config: PositionOnExchangeExecutorConfig,
        update_interval: float = 1.0,
        max_retries: int = 10,
    ):
        """
        Initialize the PositionOnExchangeExecutor instance.

        :param strategy: The strategy to be used by the PositionOnExchangeExecutor.
        :param config: The configuration for the PositionOnExchangeExecutor.
        :param update_interval: The interval at which the executor should be updated, defaults to 1.0.
        :param max_retries: The maximum number of retries, defaults to 10.
        """
        if config.triple_barrier_config.time_limit_order_type != OrderType.MARKET:
            error = "Only market orders are supported for time_limit"
            self.logger().error(error)
            raise ValueError(error)
        if config.triple_barrier_config.stop_loss_order_type not in (
            OrderType.STOP_LOSS,
            OrderType.STOP_LOSS_LIMIT,
        ):
            error = (
                f"PositionOnExchangeExecutor requires STOP_LOSS or STOP_LOSS_LIMIT order type, "
                f"got {config.triple_barrier_config.stop_loss_order_type}"
            )
            self.logger().error(error)
            raise ValueError(error)

        # Temporarily set stop_loss_order_type to MARKET to pass parent __init__
        # validation, then restore the exchange-native type.
        original_sl_type = config.triple_barrier_config.stop_loss_order_type
        config.triple_barrier_config.stop_loss_order_type = OrderType.MARKET
        super().__init__(strategy=strategy, config=config, update_interval=update_interval, max_retries=max_retries)
        config.triple_barrier_config.stop_loss_order_type = original_sl_type

        self._stop_loss_order: TrackedOrder | None = None
        self._take_profit_order: TrackedOrder | None = None

    @property
    def trade_pnl_pct(self):
        """
        Override to return 0 when a close order has been placed but not yet filled.

        For exchange-native SL/TP executors the actual execution price is only
        known after the exchange-side order fills.  Using current_market_price as
        a proxy while the order is in-flight (close_type set, _close_order pending)
        gives misleading non-zero PnL.  Return 0 until the close order is done.
        """
        if self._close_order and not self._close_order.is_done:
            return Decimal("0")
        return super().trade_pnl_pct

    @property
    def stop_loss_price(self):
        """
        Calculate the stop loss price based on entry price and configured stop loss percentage.

        :return: The stop loss price.
        """
        if self.config.side == TradeType.BUY:
            stop_loss_price = self.entry_price * (1 - self.config.triple_barrier_config.stop_loss)
        else:
            stop_loss_price = self.entry_price * (1 + self.config.triple_barrier_config.stop_loss)
        return stop_loss_price

    def control_stop_loss(self):
        """
        Override client-side stop loss monitoring: place an exchange-native stop-loss order
        instead of checking price thresholds on each control tick.

        :return: None
        """
        if self.config.triple_barrier_config.stop_loss and not self._stop_loss_order:
            self.place_stop_loss_order()

    def control_take_profit(self):
        """
        Override client-side take profit monitoring: place an exchange-native take-profit order
        or a limit order on the exchange. Handles activation bounds for limit-type orders.

        :return: None
        """
        if self.config.triple_barrier_config.take_profit and not self._take_profit_order:
            if self.config.triple_barrier_config.take_profit_order_type.is_limit_type():
                is_within_activation_bounds = self._is_within_activation_bounds(
                    self.take_profit_price,
                    self.close_order_side,
                    self.config.triple_barrier_config.take_profit_order_type,
                )
                if not self._take_profit_limit_order:
                    if is_within_activation_bounds:
                        self.place_take_profit_limit_order()
                else:
                    if (
                        self._take_profit_limit_order.is_open
                        and not self._take_profit_limit_order.is_filled
                        and not is_within_activation_bounds
                    ):
                        self.cancel_take_profit()
            else:
                self.place_take_profit_order()

    def place_stop_loss_order(self):
        """
        Place an exchange-native stop-loss order.

        Uses stop_loss_order_type from the triple barrier config (STOP_LOSS or
        STOP_LOSS_LIMIT).  For both variants the price passed to the connector is
        stop_loss_price; STOP_LOSS treats it as the trigger price while
        STOP_LOSS_LIMIT treats it as the limit price (the fill price cap at the
        stop trigger level).

        :return: None
        """
        if not self._stop_loss_order:
            order_id = self.place_order(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                amount=self.amount_to_close,
                price=self.stop_loss_price,
                order_type=self.config.triple_barrier_config.stop_loss_order_type,
                position_action=PositionAction.CLOSE,
                side=self.close_order_side,
            )
            self._stop_loss_order = TrackedOrder(order_id=order_id)
            self.logger().debug(f"Executor ID: {self.config.id} - Placing stop loss order {order_id}")
        else:
            self.logger().debug(f"Executor ID: {self.config.id} - Stop loss order already placed")

    def place_take_profit_order(self):
        """
        Place an exchange-native take-profit order.

        :return: None
        """
        if not self._take_profit_order and not self._take_profit_limit_order:
            order_id = self.place_order(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                amount=self.amount_to_close,
                price=self.take_profit_price,
                order_type=OrderType.TAKE_PROFIT,
                position_action=PositionAction.CLOSE,
                side=self.close_order_side,
            )
            self._take_profit_order = TrackedOrder(order_id=order_id)
            self.logger().debug(f"Executor ID: {self.config.id} - Placing take profit order {order_id}")
        elif self._take_profit_limit_order:
            raise ValueError("Take profit limit order attempt while a limit order is already in place")
        else:
            self.logger().debug(f"Executor ID: {self.config.id} - Take profit order already placed")

    def renew_stop_loss_order(self):
        """
        Cancel and re-place the stop loss order (e.g., after partial fills change the amount).

        :return: None
        """
        self.cancel_stop_loss()
        self._stop_loss_order = None
        self.place_stop_loss_order()
        self.logger().debug("Renewing stop loss order")

    def renew_take_profit_order(self):
        """
        Cancel and re-place the take profit order.

        :return: None
        """
        self.cancel_take_profit()
        self._take_profit_order = None
        if self._take_profit_limit_order:
            self.place_take_profit_limit_order()
        else:
            self.place_take_profit_order()
        self.logger().debug("Renewing take profit order")

    def cancel_stop_loss(self):
        """
        Cancel the exchange-native stop-loss order.

        :return: None
        """
        self._strategy.cancel(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            order_id=self._stop_loss_order.order_id,
        )
        self.logger().debug("Removing stop loss")

    def cancel_open_orders(self):
        """
        Cancel all open orders including exchange-native SL/TP orders.

        :return: None
        """
        super().cancel_open_orders()
        if self._take_profit_order and self._take_profit_order.order and self._take_profit_order.order.is_open:
            self.cancel_take_profit()
        if self._stop_loss_order and self._stop_loss_order.order and self._stop_loss_order.order.is_open:
            self.cancel_stop_loss()

    def cancel_take_profit(self):
        """
        Cancel the exchange-native take-profit order.

        :return: None
        """
        self._strategy.cancel(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            order_id=self._take_profit_order.order_id,
        )
        self.logger().debug("Removing take profit")

    def cancel_open_order(self):
        """
        Cancel the open (entry) order.

        :return: None
        """
        self._strategy.cancel(
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            order_id=self._open_order.order_id,
        )
        self.logger().debug("Removing open order")

    def update_tracked_orders_with_order_id(self, order_id: str):
        """
        Update tracked orders with InFlightOrder information, including SL/TP orders.

        :param order_id: The order_id to be used as a reference.
        :return: None
        """
        in_flight_order = self.get_in_flight_order(self.config.connector_name, order_id)
        super().update_tracked_orders_with_order_id(order_id)

        if self._stop_loss_order and self._stop_loss_order.order_id == order_id:
            self._stop_loss_order.order = in_flight_order
        if self._take_profit_order and self._take_profit_order.order_id == order_id:
            self._take_profit_order.order = in_flight_order

    def process_order_completed_event(self, _, market, event: BuyOrderCompletedEvent | SellOrderCompletedEvent):
        """
        Process order completed events. Check if a SL/TP order was filled and update state.
        """
        self._total_executed_amount_backup += event.base_asset_amount
        self.update_tracked_orders_with_order_id(event.order_id)
        super().process_order_completed_event(_, market, event)

        if self._stop_loss_order and self._stop_loss_order.order_id == event.order_id:
            self.close_type = CloseType.STOP_LOSS
            self._close_order = self._stop_loss_order
            self._status = RunnableStatus.SHUTTING_DOWN
        if self._take_profit_order and self._take_profit_order.order_id == event.order_id:
            self.close_type = CloseType.TAKE_PROFIT
            self._close_order = self._take_profit_order
            self._status = RunnableStatus.SHUTTING_DOWN

    def process_order_canceled_event(self, _, market: ConnectorBase, event: OrderCancelledEvent):
        """
        Process order canceled events for SL/TP orders.
        """
        super().process_order_canceled_event(_, market, event)
        if self._stop_loss_order and event.order_id == self._stop_loss_order.order_id:
            self._failed_orders.append(self._stop_loss_order)
            self._stop_loss_order = None
        if self._take_profit_order and event.order_id == self._take_profit_order.order_id:
            self._failed_orders.append(self._take_profit_order)
            self._take_profit_order = None

    def process_order_failed_event(self, _, market, event: MarketOrderFailureEvent):
        """
        Process order failed events. Re-attempt SL/TP placement on next control tick.
        """
        super().process_order_failed_event(_, market, event)
        if self._stop_loss_order and event.order_id == self._stop_loss_order.order_id:
            self._failed_orders.append(self._stop_loss_order)
            self._stop_loss_order = None
            self.logger().error(
                f"Stop loss order failed {event.order_id}. Retrying {self._current_retries}/{self._max_retries}"
            )
        if self._take_profit_order and event.order_id == self._take_profit_order.order_id:
            self._failed_orders.append(self._take_profit_order)
            self._take_profit_order = None
            self.logger().error(
                f"Take profit order failed {event.order_id}. Retrying {self._current_retries}/{self._max_retries}"
            )

    async def _sleep(self, delay: float):
        """
        Sleep helper for shutdown process.

        :param delay: The time to sleep.
        :return: None
        """
        await asyncio.sleep(delay)
