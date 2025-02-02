import logging
from decimal import Decimal
from math import floor
from typing import Dict, List

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate, PerpetualOrderCandidate
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.strategy_v2.executors.executor_base import ExecutorBase
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder

from .control_mixin import ControlMixin
from .data_types import ProgressiveExecutorConfig, ProgressiveExecutorUpdates
from .order_execution_mixin import OrderExecutionMixin
from .order_management_mixin import OrderManagementMixin
from .order_processing_mixin import OrderProcessingMixin
from .pnl_calculator_mixin import PNLCalculatorMixin


class ProgressiveExecutor(
    OrderManagementMixin,
    OrderExecutionMixin,
    OrderProcessingMixin,
    PNLCalculatorMixin,
    ControlMixin,
    ExecutorBase,
):
    _logger = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(self, strategy: ScriptStrategyBase, config: ProgressiveExecutorConfig,
                 update_interval: float = 1.0, max_retries: int = 10):
        """
        Initialize the ProgressiveExecutor instance.

        :param strategy: The strategy to be used by the ProgressiveExecutor.
        :param config: The configuration for the ProgressiveExecutor, subclass of PositionExecutorConfig.
        :param update_interval: The interval at which the ProgressiveExecutor should be updated, defaults to 1.0.
        :param max_retries: The maximum number of retries for the ProgressiveExecutor, defaults to 5.
        """
        if config.triple_barrier_config.time_limit_order_type != OrderType.MARKET or \
                config.triple_barrier_config.stop_loss_order_type != OrderType.MARKET:
            error = "Only market orders are supported for time_limit and stop_loss"
            self.logger().error(error)
            raise ValueError(error)
        super().__init__(
            strategy=strategy,
            config=config,
            connectors=[config.connector_name],
            update_interval=update_interval,
        )
        if not config.entry_price:
            open_order_price_type = PriceType.BestBid if config.side == TradeType.BUY else PriceType.BestAsk
            config.entry_price = self.get_price(config.connector_name, config.trading_pair,
                                                price_type=open_order_price_type)
        self.config: ProgressiveExecutorConfig = config

        self._open_order_timestamp = None

        # Order tracking
        self._trailing_stop_trigger_pct: Decimal | None = None
        self._open_order: TrackedOrder | None = None
        self._close_order: TrackedOrder | None = None
        self._realized_orders: List[TrackedOrder] = []
        self._failed_orders: List[TrackedOrder] = []
        self._canceled_orders: List[TrackedOrder] = []

        self._total_executed_amount_backup: Decimal = Decimal("0")
        self._current_retries = 0
        self._max_retries = max_retries

    @property
    def strategy(self) -> ScriptStrategyBase:
        """
        :return: The strategy instance.
        """
        return self._strategy

    @property
    def open_order(self) -> TrackedOrder:
        return self._open_order

    @open_order.setter
    def open_order(self, value: TrackedOrder):
        self._open_order = value

    @property
    def close_order(self) -> TrackedOrder:
        return self._close_order

    @close_order.setter
    def close_order(self, value: TrackedOrder):
        self._close_order = value

    @property
    def realized_orders(self) -> List[TrackedOrder]:
        return self._realized_orders

    @realized_orders.setter
    def realized_orders(self, value: List[TrackedOrder]):
        self._realized_orders = value

    @property
    def failed_orders(self) -> List[TrackedOrder]:
        return self._failed_orders

    @failed_orders.setter
    def failed_orders(self, value: List[TrackedOrder]):
        self._failed_orders = value

    @property
    def canceled_orders(self) -> List[TrackedOrder]:
        return self._failed_orders

    @canceled_orders.setter
    def canceled_orders(self, value: List[TrackedOrder]):
        self._failed_orders = value

    @property
    def open_order_timestamp(self) -> float | int | None:
        return self._open_order_timestamp

    @open_order_timestamp.setter
    def open_order_timestamp(self, value: float | int | None):
        self._open_order_timestamp = value

    @property
    def trailing_stop_trigger_pnl(self) -> Decimal | None:
        return self._trailing_stop_trigger_pct

    @trailing_stop_trigger_pnl.setter
    def trailing_stop_trigger_pnl(self, value: Decimal):
        self._trailing_stop_trigger_pct = value

    @property
    def current_retries(self) -> int:
        return self._current_retries

    @current_retries.setter
    def current_retries(self, value: int):
        self._current_retries = value

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def total_executed_amount_backup(self) -> Decimal:
        return self._total_executed_amount_backup

    @total_executed_amount_backup.setter
    def total_executed_amount_backup(self, value: Decimal):
        self._total_executed_amount_backup = value

    @property
    def is_perpetual(self) -> bool:
        """
        Check if the exchange connector is perpetual.

        :return: True if the exchange connector is perpetual, False otherwise.
        """
        return self.is_perpetual_connector(self.config.connector_name)

    @property
    def is_trading(self):
        """
        Check if the position is trading.

        :return: True if the position is trading, False otherwise.
        """
        return self.status == RunnableStatus.RUNNING and self.open_filled_amount > Decimal("0")

    @property
    def is_expired(self) -> bool:
        """
        Check if the position is expired.

        :return: True if the position is expired, False otherwise.
        """
        return self.end_time and (self.end_time <= self.current_timestamp)

    @property
    def is_extended_on_yield(self) -> bool:
        """
        Check if the position is extended.

        :return: True if the position is extended for PnL above APR, False otherwise.
        """
        return self.is_expired and (self.get_net_pnl_pct() > self.get_target_pnl_yield())

    @property
    def current_market_price(self) -> Decimal:
        """
        This method is responsible for getting the current market price to be used as a reference for control barriers.

        :return: The current market price.
        """
        price_type = PriceType.BestAsk if self.config.side == TradeType.BUY else PriceType.BestBid
        return self.get_price(self.config.connector_name, self.config.trading_pair, price_type=price_type)

    @property
    def current_timestamp(self) -> float:
        """
        :return: The current timestamp.
        """
        return self.strategy.current_timestamp

    @property
    def entry_price(self) -> Decimal:
        """
        This method is responsible for getting the entry price. If the open order is done, it returns the average executed price.
        If the entry price is set in the configuration, it returns the entry price from the configuration.
        Otherwise, it returns the best ask price for buy orders and the best bid price for a sell orders.

        :return: The entry price.
        """
        if self.open_order and self.open_order.is_done:
            return self.open_order.average_executed_price
        elif self.config.triple_barrier_config.open_order_type == OrderType.LIMIT_MAKER:
            if self.config.side == TradeType.BUY:
                best_bid = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestBid)
                return min(self.config.entry_price, best_bid)
            else:
                best_ask = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.BestAsk)
                return max(self.config.entry_price, best_ask)
        else:
            return self.config.entry_price

    @property
    def close_price(self) -> Decimal:
        """
        This method is responsible for getting the close price. If the close order is done, it returns the average executed price.
        Otherwise, it returns the current market price.

        :return: The close price.
        """
        if self.close_order and self.close_order.is_done:
            return self.close_order.average_executed_price
        else:
            return self.current_market_price

    @property
    def side(self) -> TradeType:
        """
        This method is responsible for getting the trade type.

        :return: The trade type.
        """
        return self.config.side

    @property
    def end_time(self) -> float | None:
        """
        Calculate the end time of the position based on the time limit

        :return: The end time of the position.
        """
        if not self.config.triple_barrier_config.time_limit:
            return None
        return self.config.timestamp + self.config.triple_barrier_config.time_limit

    def evaluate_max_retries(self):
        """
        This method is responsible for evaluating the maximum number of retries to place an order and stop the executor
        if the maximum number of retries is reached.

        :return: None
        """
        if self.current_retries > self.max_retries:
            self.close_type = CloseType.FAILED
            self.stop()

    async def on_start(self):
        """
        This method is responsible for starting the executor and validating if the position is expired. The base method
        validates if there is enough balance to place the open order.

        :return: None
        """
        self.logger().debug("Starting ProgressiveExecutor")
        await super().on_start()
        if self.is_expired:
            self.close_type = CloseType.EXPIRED
            self.stop()

    def _is_within_activation_bounds(self, close_price: Decimal) -> bool:
        """
        This method is responsible for checking if the close price is within the activation bounds to place the open
        order. If the activation bounds are not set, it returns True. This makes the executor more capital efficient.

        :param close_price: The close price to be checked.
        :return: True if the close price is within the activation bounds, False otherwise.
        """
        activation_bounds = self.config.activation_bounds
        order_price = self.config.entry_price
        if not activation_bounds:
            return True
        if self.config.triple_barrier_config.open_order_type == OrderType.LIMIT:
            return (
                order_price > close_price * (1 - activation_bounds[0])
                if self.config.side == TradeType.BUY
                else order_price < close_price * (1 + activation_bounds[0])
            )
        if self.config.side == TradeType.BUY:
            return order_price < close_price * (1 - activation_bounds[1])
        else:
            return order_price > close_price * (1 + activation_bounds[1])

    def update_live(self, update_data: ProgressiveExecutorUpdates):
        """
        This method allows strategy to stop the executor early.

        :return: None
        """
        self.config.triple_barrier_config = self.config.triple_barrier_config.new_instance_with_adjusted_volatility(
            abs(float(update_data.volatility * Decimal("1.1") / self.config.triple_barrier_config.stop_loss))
        )
        self.logger().error(f"Updating executor with data: {update_data}")

    def get_custom_info(self) -> Dict:
        return {
            "level_id": self.config.level_id,
            "current_position_average_price": self.entry_price,
            "side": self.config.side,
            "current_retries": self.current_retries,
            "max_retries": self.max_retries
        }

    def to_format_status(self, scale=1.0) -> List[str]:
        lines = []
        current_price = self.get_price(self.config.connector_name, self.config.trading_pair)
        # amount_in_quote = self.entry_price * (
        #     self.open_filled_amount if self.open_filled_amount > Decimal("0") else self.config.amount)
        # quote_asset = self.config.trading_pair.split("-")[1]
        # if not self.is_closed:
        #     lines.extend([
        #         f"{'=' * 10}"
        #         # f"| Trading Pair: {self.config.trading_pair} | Exchange: {self.config.connector_name} | Side: {self.config.side} "
        #         # f"| Entry price: {self.entry_price:.2g} | Close price: {self.close_price:.2g} | Amount: {amount_in_quote:.4f} {quote_asset} "
        #         f" | Realized PNL: {self.trade_pnl_quote:.2g} {quote_asset} | Total Fee: {self.cum_fees_quote:.2g} {quote_asset} "
        #         f"| PNL (%): {self.net_pnl_pct * 100:.2f}% | PNL (abs): {self.net_pnl_quote:.2g} {quote_asset} | Close Type: {self.close_type} |"
        #     ])
        # else:
        #     lines.extend([
        #         f"{'=' * 10}"
        #         # f"| Trading Pair: {self.config.trading_pair} | Exchange: {self.config.connector_name} | Side: {self.config.side} "
        #         # f"| Entry price: {self.entry_price:.2g} | Close price: {self.close_price:.2g} | Amount: {amount_in_quote:.4f} {quote_asset} "
        #         f"| Unrealized PNL: {self.trade_pnl_quote:.2g} {quote_asset} | Total Fee: {self.cum_fees_quote:.2g} {quote_asset} "
        #         f"| PNL (%): {self.net_pnl_pct * 100:.2f}% | PNL (abs): {self.net_pnl_quote:.2g} {quote_asset} | Close Type: {self.close_type} |"
        #     ])
        if self.is_trading:
            # lines.extend([f"{'=' * 10}"])
            if self.config.triple_barrier_config.time_limit:
                time_scale = int(scale * 200)
                seconds_remaining = (self.end_time - self.current_timestamp)
                time_progress = (self.config.triple_barrier_config.time_limit - seconds_remaining) / self.config.triple_barrier_config.time_limit
                time_bar = "".join(['-' if i < time_scale * time_progress else ' ' for i in range(time_scale)])
                lines.extend([f"{self.config.trading_pair:>10} - Time limit:    [{time_bar}]"])

            progress = 0
            if self.config.triple_barrier_config.stop_loss:
                price_scale = int(scale * 200)
                stop_loss_price = self.entry_price * (1 - self.config.triple_barrier_config.stop_loss) if self.config.side == TradeType.BUY \
                    else self.entry_price * (1 + self.config.triple_barrier_config.stop_loss)
                take_profit_pct: Decimal = Decimal("0.2")
                take_profit_price = self.entry_price * Decimal(1 + take_profit_pct) if self.config.side == TradeType.BUY \
                    else self.entry_price * Decimal(1 - min(take_profit_pct, Decimal("1")))

                trailing_stop_price = None
                if self.config.side == TradeType.BUY:
                    price_range = take_profit_price - stop_loss_price
                    progress = (current_price - stop_loss_price) / price_range
                    entry_price = (self.entry_price - stop_loss_price) / price_range
                    if self.trailing_stop_trigger_pnl:
                        trailing_stop_price = (self.entry_price * (1 + self.trade_pnl_pct - self.trailing_stop_trigger_pnl) - stop_loss_price) / price_range
                elif self.config.side == TradeType.SELL:
                    price_range = stop_loss_price - take_profit_price
                    progress = (stop_loss_price - current_price) / price_range
                    entry_price = (stop_loss_price - self.entry_price) / price_range
                    if self.trailing_stop_trigger_pnl:
                        trailing_stop_price = (stop_loss_price - self.entry_price * (1 + self.trailing_stop_trigger_pnl - self.trade_pnl_pct)) / price_range
                else:
                    entry_price = 0
                    price_range = 1

                # progress_bar = [f'|' if i == int(price_scale * progress) else ' ' for i in range(price_scale)]

                zero = int(price_scale * entry_price)
                progress_index = int(price_scale * progress)

                progress_bar = ['.' for _ in range(price_scale)]
                progress_bar[progress_index] = '|'
                if progress_index < zero:
                    progress_bar[progress_index + 1:zero] = ['<'] * (abs(zero - progress_index) - 1)
                if progress_index > zero:
                    progress_bar[zero + 1:progress_index] = ['>'] * (abs(zero - progress_index) - 1)

                price_bar = [' ' for _ in range(price_scale)]
                trailing_bar = [' ' for _ in range(price_scale)]
                price_bar.insert(0, f"{' ' * 1}")
                price_bar.append(f"{' ' * 6}")
                trailing_bar.append(f"{' ' * 1}")

                if 0 <= zero < len(progress_bar):
                    progress_bar[zero] = '*'
                    trailing_bar[zero] = '0'
                    trailing_bar[zero - 1:zero + 1] = f'{0:3}%'

                if self.config.triple_barrier_config.trailing_stop:
                    for i, (pct, r) in enumerate(self.config.triple_barrier_config.trailing_stop.take_profit_table):
                        if 0 < pct <= take_profit_pct:
                            if self.config.side == TradeType.BUY:
                                position = int(price_scale * ((1 + pct) * self.entry_price - stop_loss_price) / price_range)
                            else:
                                position = int(price_scale * (stop_loss_price - (1 - pct) * self.entry_price) / price_range)

                            if 0 <= position < len(progress_bar):
                                progress_bar[position] = '*'
                                trailing_bar[position:position + 3] = f'{floor(r * Decimal("99.99")):3}%'

                progress_bar[int(price_scale * progress)] = "|"
                if trailing_stop_price:
                    progress_bar[int(price_scale * trailing_stop_price)] = "%"
                pnl_pct: Decimal = self.trade_pnl_pct * 100
                if progress > zero:
                    price_bar[int(price_scale * progress):int(price_scale * progress) + 7] = f'{pnl_pct:+.1f}% ({current_price:.3g})' if pnl_pct < Decimal("10") else f'{pnl_pct:+.1f}% ({current_price}:.3g)'
                else:
                    price_bar[int(price_scale * progress):int(price_scale * progress) + 7] = f'{pnl_pct:+.1f}% ({current_price:.3g})' if pnl_pct < Decimal("10") else f'{pnl_pct:+.1f}% ({current_price}:.3g)'

                sl_label: str = f"{' ' * 13}SL: {stop_loss_price:10.2f} ["
                progress_bar.insert(0, sl_label)
                progress_bar.append(f"] TP: {take_profit_price:8.2f} ({take_profit_pct * 100}% PnL {'BUY' if self.config.side == TradeType.BUY else 'SELL'})")
                price_bar.insert(0, f"{' ' * (len(sl_label) - 1)}")
                trailing_bar.insert(0, f"{' ' * (len(sl_label) - 1)}")
                lines.extend([f"{''.join(progress_bar)}"])
                lines.extend([f"{''.join(price_bar)}"])
                lines.extend([f"{''.join(trailing_bar)}"])

            if self.trailing_stop_trigger_pnl:
                lines.extend([f"             Trailing stop pnl trigger: {self.trailing_stop_trigger_pnl:.2g}"])

            # lines.extend([f"{' ' * 10}"])
        return lines

    async def validate_sufficient_balance(self):
        if self.is_perpetual:
            order_candidate = PerpetualOrderCandidate(
                trading_pair=self.config.trading_pair,
                is_maker=self.config.triple_barrier_config.open_order_type.is_limit_type(),
                order_type=self.config.triple_barrier_config.open_order_type,
                order_side=self.config.side,
                amount=self.config.amount,
                price=self.entry_price,
                leverage=Decimal(self.config.leverage),
            )
        else:
            order_candidate = OrderCandidate(
                trading_pair=self.config.trading_pair,
                is_maker=self.config.triple_barrier_config.open_order_type.is_limit_type(),
                order_type=self.config.triple_barrier_config.open_order_type,
                order_side=self.config.side,
                amount=self.config.amount,
                price=self.entry_price,
            )
        adjusted_order_candidates = self.adjust_order_candidates(self.config.connector_name, [order_candidate])
        if adjusted_order_candidates[0].amount == Decimal("0"):
            self.close_type = CloseType.INSUFFICIENT_BALANCE
            self.logger().warning(f"Not enough budget to open a position. {self.config.trading_pair}:{self.config.side}:{self.config.amount}")
            self.stop()
