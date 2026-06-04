import asyncio
import time
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, Union

from hummingbot.client.config.trade_fee_schema_loader import TradeFeeSchemaLoader
from hummingbot.connector.constants import s_decimal_0, s_decimal_NaN
from hummingbot.connector.in_flight_order_base import InFlightOrderBase
from hummingbot.connector.utils import TradeFillOrderDetails, split_hb_trading_pair
from hummingbot.core.data_type.cancellation_result import CancellationResult
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.data_type.market_order import MarketOrder
from hummingbot.core.event.event_logger import EventLogger
from hummingbot.core.event.event_reporter import EventReporter
from hummingbot.core.event.events import MarketEvent, OrderFilledEvent
from hummingbot.core.network_iterator import NetworkIterator
from hummingbot.core.utils.estimate_fee import estimate_fee


class ConnectorBase(NetworkIterator):
    MARKET_EVENTS = [
        MarketEvent.ReceivedAsset,
        MarketEvent.BuyOrderCompleted,
        MarketEvent.SellOrderCompleted,
        MarketEvent.WithdrawAsset,
        MarketEvent.OrderCancelled,
        MarketEvent.OrderFilled,
        MarketEvent.OrderExpired,
        MarketEvent.OrderFailure,
        MarketEvent.TransactionFailure,
        MarketEvent.BuyOrderCreated,
        MarketEvent.SellOrderCreated,
        MarketEvent.FundingPaymentCompleted,
        MarketEvent.RangePositionLiquidityAdded,
        MarketEvent.RangePositionLiquidityRemoved,
        MarketEvent.RangePositionUpdateFailure,
    ]

    def __new__(cls, *args, **kwargs):
        # Mirrors Cython C-level allocation. Account/balance/order tracking
        # dicts are accessed from event handlers during live dispatch (e.g.,
        # by the same chain that surfaced AttributeError in OkxExchange).
        instance = super().__new__(cls)
        instance._account_balances = {}
        instance._account_available_balances = {}
        instance._real_time_balance_update = True
        instance._in_flight_orders_snapshot = {}
        instance._in_flight_orders_snapshot_timestamp = 0.0
        instance._current_trade_fills = set()
        instance._exchange_order_ids = {}
        instance._trade_fee_schema = None
        return instance

    def __init__(self, balance_asset_limit: Optional[Dict[str, Dict[str, Decimal]]] = None):
        super().__init__()

        self._event_reporter = EventReporter(event_source=self.display_name)
        self._event_logger = EventLogger(event_source=self.display_name)
        for event_tag in self.MARKET_EVENTS:
            self.add_listener(event_tag, self._event_reporter)
            self.add_listener(event_tag, self._event_logger)

        self._account_balances: Dict[str, Decimal] = {}
        self._account_available_balances: Dict[str, Decimal] = {}
        # _real_time_balance_update is used to flag whether the connector provides real time balance updates.
        # if not, the available will be calculated based on what happened since snapshot taken.
        self._real_time_balance_update: bool = True
        # If _real_time_balance_update is set to False, Sub classes of this connector class need to set values
        # for _in_flight_orders_snapshot and _in_flight_orders_snapshot_timestamp when the update user balances.
        self._in_flight_orders_snapshot: Dict[str, InFlightOrderBase] = {}
        self._in_flight_orders_snapshot_timestamp: float = 0.0
        self._current_trade_fills: Set[TradeFillOrderDetails] = set()
        self._exchange_order_ids: Dict[str, str] = dict()
        self._trade_fee_schema = None
        self._balance_asset_limit: Dict[str, Dict[str, object]] = balance_asset_limit or dict()

    # -- cdef → Python method conversions (Variant C: Python wrapper, no recursion risk) --

    def c_tick(self, timestamp: float) -> None:
        # Variant B: parent still Cython — call unbound c_tick via NetworkIterator
        NetworkIterator.c_tick(self, timestamp)
        self.tick(timestamp)

    def c_start(self, clock, timestamp: float) -> None:
        # Variant B: parent still Cython
        NetworkIterator.c_start(self, clock, timestamp)

    def c_stop(self, clock) -> None:
        # Variant B: parent still Cython
        NetworkIterator.c_stop(self, clock)

    def c_buy(
        self,
        trading_pair: str,
        amount: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Decimal = s_decimal_NaN,
        **kwargs,
    ) -> str:
        return self.buy(trading_pair, amount, order_type, price, **kwargs)

    def c_sell(
        self,
        trading_pair: str,
        amount: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Decimal = s_decimal_NaN,
        **kwargs,
    ) -> str:
        return self.sell(trading_pair, amount, order_type, price, **kwargs)

    def c_cancel(self, trading_pair: str, client_order_id: str) -> None:
        self.cancel(trading_pair, client_order_id)

    def c_get_balance(self, currency: str) -> Decimal:
        return self.get_balance(currency)

    def c_get_available_balance(self, currency: str) -> Decimal:
        return self.get_available_balance(currency)

    def c_get_price(self, trading_pair: str, is_buy: bool) -> Decimal:
        return self.get_price(trading_pair, is_buy)

    def c_get_order_price_quantum(self, trading_pair: str, price: Decimal) -> Decimal:
        return self.get_order_price_quantum(trading_pair, price)

    def c_get_order_size_quantum(self, trading_pair: str, order_size: Decimal) -> Decimal:
        return self.get_order_size_quantum(trading_pair, order_size)

    def c_quantize_order_price(self, trading_pair: str, price: Decimal) -> Decimal:
        if price.is_nan():
            return price
        price_quantum = self.c_get_order_price_quantum(trading_pair, price)
        return (price // price_quantum) * price_quantum

    def c_quantize_order_amount(self, trading_pair: str, amount: Decimal, price: Decimal = s_decimal_NaN) -> Decimal:
        order_size_quantum = self.c_get_order_size_quantum(trading_pair, amount)
        return (amount // order_size_quantum) * order_size_quantum

    # -- Properties --

    @property
    def real_time_balance_update(self) -> bool:
        return self._real_time_balance_update

    @real_time_balance_update.setter
    def real_time_balance_update(self, value: bool) -> None:
        self._real_time_balance_update = value

    @property
    def in_flight_orders_snapshot(self) -> Dict[str, InFlightOrderBase]:
        return self._in_flight_orders_snapshot

    @in_flight_orders_snapshot.setter
    def in_flight_orders_snapshot(self, value: Dict[str, InFlightOrderBase]) -> None:
        self._in_flight_orders_snapshot = value

    @property
    def in_flight_orders_snapshot_timestamp(self) -> float:
        return self._in_flight_orders_snapshot_timestamp

    @in_flight_orders_snapshot_timestamp.setter
    def in_flight_orders_snapshot_timestamp(self, value: float) -> None:
        self._in_flight_orders_snapshot_timestamp = value

    @property
    def status_dict(self) -> Dict[str, bool]:
        """A dictionary of statuses of various connector's components."""
        raise NotImplementedError

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def event_logs(self) -> List[object]:
        return self._event_logger.event_log

    @property
    def ready(self) -> bool:
        """Indicates whether the connector is ready to be used."""
        raise NotImplementedError

    @property
    def in_flight_orders(self) -> Dict[str, InFlightOrderBase]:
        raise NotImplementedError

    @property
    def tracking_states(self) -> Dict[str, object]:
        return {}

    @property
    def available_balances(self) -> Dict[str, Decimal]:
        return self._account_available_balances

    # -- Public methods --

    def estimate_fee_pct(self, is_maker: bool) -> Decimal:
        """
        Estimate the trading fee for maker or taker type of order.
        :param is_maker: Whether to get trading for maker or taker order
        :returns An estimated fee in percentage value
        """
        return estimate_fee(self.name, is_maker).percent

    @staticmethod
    def split_trading_pair(trading_pair: str) -> Tuple[str, str]:
        return split_hb_trading_pair(trading_pair)

    def restore_tracking_states(self, saved_states: Dict[str, object]) -> None:
        """
        Restores the tracking states from a previously saved state.
        :param saved_states: Previously saved tracking states from `tracking_states` property.
        """

    def tick(self, timestamp: float) -> None:
        """Is called automatically by the clock for each clock's tick (1 second by default)."""

    def start(self, clock, timestamp: float) -> None:
        NetworkIterator.c_start(self, clock, timestamp)

    def in_flight_asset_balances(self, in_flight_orders: Dict[str, InFlightOrderBase]) -> Dict[str, Decimal]:
        """
        Calculates total asset balances locked in in_flight_orders including fee (estimated).
        For BUY order, this is the quote asset balance locked in the order.
        For SELL order, this is the base asset balance locked in the order.
        :param in_flight_orders: a dictionary of in-flight orders
        :return A dictionary of tokens and their balance locked in the orders
        """
        asset_balances: Dict[str, Decimal] = {}
        if in_flight_orders is None:
            return asset_balances
        for order in (o for o in in_flight_orders.values() if not (o.is_done or o.is_failure or o.is_cancelled)):
            outstanding_amount = order.amount - order.executed_amount_base
            if order.trade_type is TradeType.BUY:
                outstanding_value = outstanding_amount * order.price
                if order.quote_asset not in asset_balances:
                    asset_balances[order.quote_asset] = s_decimal_0
                fee = self.estimate_fee_pct(True)
                outstanding_value *= Decimal(1) + fee
                asset_balances[order.quote_asset] += outstanding_value
            else:
                if order.base_asset not in asset_balances:
                    asset_balances[order.base_asset] = s_decimal_0
                asset_balances[order.base_asset] += outstanding_amount
        return asset_balances

    def order_filled_balances(self, starting_timestamp: float = 0) -> Dict[str, Decimal]:
        """
        Calculates total asset balance changes from filled orders since the timestamp.
        For BUY filled order, the quote balance goes down while the base balance goes up, and for SELL order,
        it's the opposite. This does not account for fee.
        :param starting_timestamp: The starting timestamp to include filter order filled events
        :returns A dictionary of tokens and their balance
        """
        order_filled_events = list(filter(lambda e: isinstance(e, OrderFilledEvent), self.event_logs))
        order_filled_events = [o for o in order_filled_events if o.timestamp > starting_timestamp]
        balances: Dict[str, Decimal] = {}
        for event in order_filled_events:
            base, quote = event.trading_pair.split("-")[0], event.trading_pair.split("-")[1]
            if event.trade_type is TradeType.BUY:
                quote_value = Decimal("-1") * event.price * event.amount
                base_value = event.amount
            else:
                quote_value = event.price * event.amount
                base_value = Decimal("-1") * event.amount
            if base not in balances:
                balances[base] = s_decimal_0
            if quote not in balances:
                balances[quote] = s_decimal_0
            balances[base] += base_value
            balances[quote] += quote_value
        return balances

    def get_exchange_limit_config(self, market: str) -> Dict[str, object]:
        """Retrieves the Balance Limits for the specified market."""
        exchange_limits = self._balance_asset_limit.get(market, {})
        return exchange_limits if exchange_limits is not None else {}

    async def cancel_all(self, timeout_seconds: float) -> List[CancellationResult]:
        """
        Cancels all in-flight orders and waits for cancellation results.
        Used by bot's top level stop and exit commands (cancelling outstanding orders on exit).
        :param timeout_seconds: The timeout at which the operation will be canceled.
        :returns List of CancellationResult which indicates whether each order is successfully canceled.
        """
        raise NotImplementedError

    def buy(self, trading_pair: str, amount: Decimal, order_type: OrderType, price: Decimal, **kwargs) -> str:
        """
        Buys an amount of base asset (of the given trading pair).
        :param trading_pair: The market (e.g. BTC-USDT) to buy from
        :param amount: The amount in base token value
        :param order_type: The order type
        :param price: The price (note: this is no longer optional)
        :returns An order id
        """
        raise NotImplementedError

    def sell(self, trading_pair: str, amount: Decimal, order_type: OrderType, price: Decimal, **kwargs) -> str:
        """
        Sells an amount of base asset (of the given trading pair).
        :param trading_pair: The market (e.g. BTC-USDT) to sell from
        :param amount: The amount in base token value
        :param order_type: The order type
        :param price: The price (note: this is no longer optional)
        :returns An order id
        """
        raise NotImplementedError

    def batch_order_create(
        self, orders_to_create: List[Union[LimitOrder, MarketOrder]]
    ) -> List[Union[LimitOrder, MarketOrder]]:
        """
        Issues a batch order creation as a single API request for exchanges that implement this feature. The default
        implementation of this method is to send the requests discretely (one by one).
        :param orders_to_create: A list of LimitOrder or MarketOrder objects representing the orders to create. The
            order IDs can be blank.
        :returns: A list of LimitOrder or MarketOrder objects representing the created orders, complete with the
            generated order IDs.
        """
        creation_results = []
        for order in orders_to_create:
            order_type = OrderType.LIMIT if isinstance(order, LimitOrder) else OrderType.MARKET
            size = order.quantity if order_type == OrderType.LIMIT else order.amount
            if order.is_buy:
                client_order_id = self.buy(
                    trading_pair=order.trading_pair,
                    amount=size,
                    order_type=order_type,
                    price=order.price if order_type == OrderType.LIMIT else s_decimal_NaN,
                )
            else:
                client_order_id = self.sell(
                    trading_pair=order.trading_pair,
                    amount=size,
                    order_type=order_type,
                    price=order.price if order_type == OrderType.LIMIT else s_decimal_NaN,
                )
            if order_type == OrderType.LIMIT:
                creation_results.append(
                    LimitOrder(
                        client_order_id=client_order_id,
                        trading_pair=order.trading_pair,
                        is_buy=order.is_buy,
                        base_currency=order.base_currency,
                        quote_currency=order.quote_currency,
                        price=order.price,
                        quantity=size,
                        filled_quantity=order.filled_quantity,
                        creation_timestamp=order.creation_timestamp,
                        status=order.status,
                    )
                )
            else:
                creation_results.append(
                    MarketOrder(
                        order_id=client_order_id,
                        trading_pair=order.trading_pair,
                        is_buy=order.is_buy,
                        base_asset=order.base_asset,
                        quote_asset=order.quote_asset,
                        amount=size,
                        timestamp=order.timestamp,
                    )
                )
        return creation_results

    def cancel(self, trading_pair: str, client_order_id: str) -> None:
        """
        Cancel an order.
        :param trading_pair: The market (e.g. BTC-USDT) of the order.
        :param client_order_id: The internal order id (also called client_order_id)
        """
        raise NotImplementedError

    def batch_order_cancel(self, orders_to_cancel: List[LimitOrder]) -> None:
        """
        Issues a batch order cancelation as a single API request for exchanges that implement this feature. The default
        implementation of this method is to send the requests discretely (one by one).
        :param orders_to_cancel: A list of the orders to cancel.
        """
        for order in orders_to_cancel:
            self.cancel(trading_pair=order.trading_pair, client_order_id=order.client_order_id)

    def stop_tracking_order(self, order_id: str) -> None:
        """Stops tracking an in-flight order."""
        raise NotImplementedError

    def get_all_balances(self) -> Dict[str, Decimal]:
        """:return: Dict[asset_name: asset_balance]: Total balances of all assets"""
        return self._account_balances.copy()

    def get_balance(self, currency: str) -> Decimal:
        """
        :param currency: The currency (token) name
        :return: A balance for the given currency (token)
        """
        return self._account_balances.get(currency, s_decimal_0)

    def apply_balance_limit(self, currency: str, available_balance: Decimal, limit: Decimal) -> Decimal:
        """
        Apply budget limit on an available balance, the limit is calculated as followings:
        - Minus balance used in outstanding orders (in flight orders)
        - Plus balance accredited from filled orders (since the bot started)
        :param currency: The currency (token) name
        :param available_balance: The available balance of the token
        :param limit: The balance limit for the token
        :returns An available balance after the limit has been applied
        """
        in_flight_balance = self.in_flight_asset_balances(self.in_flight_orders).get(currency, s_decimal_0)
        limit -= in_flight_balance
        filled_balance = self.order_filled_balances().get(currency, s_decimal_0)
        limit += filled_balance
        limit = max(limit, s_decimal_0)
        return min(available_balance, limit)

    def apply_balance_update_since_snapshot(self, currency: str, available_balance: Decimal) -> Decimal:
        """
        Applies available balance update accounting for changes since last snapshot.
        :param currency: the token symbol
        :param available_balance: the current available_balance (snap balance taken since last _update_balances())
        :returns the real available that accounts for changes in flight orders and filled orders
        """
        snapshot_bal = self.in_flight_asset_balances(self._in_flight_orders_snapshot).get(currency, s_decimal_0)
        in_flight_bal = self.in_flight_asset_balances(self.in_flight_orders).get(currency, s_decimal_0)
        orders_filled_bal = self.order_filled_balances(self._in_flight_orders_snapshot_timestamp).get(
            currency, s_decimal_0
        )
        actual_available = available_balance + snapshot_bal - in_flight_bal + orders_filled_bal
        return actual_available

    def get_available_balance(self, currency: str) -> Decimal:
        """
        Return available balance for a given currency. The function accounts for balance changes since the last time
        the snapshot was taken if no real time balance update. The function applied limit if configured.
        :param currency: The currency (token) name
        :returns: Balance available for trading for the specified currency
        """
        available_balance = self._account_available_balances.get(currency, s_decimal_0)
        if not self._real_time_balance_update:
            available_balance = self.apply_balance_update_since_snapshot(currency, available_balance)
        balance_limits = self.get_exchange_limit_config(self.name)
        if currency in balance_limits:
            balance_limit = Decimal(str(balance_limits[currency]))
            available_balance = self.apply_balance_limit(currency, available_balance, balance_limit)
        return available_balance

    def get_price(self, trading_pair: str, is_buy: bool, amount: Decimal = s_decimal_NaN) -> Decimal:
        """
        Get price for the market trading pair.
        :param trading_pair: The market trading pair
        :param is_buy: Whether to buy or sell the underlying asset
        :param amount: The amount (to buy or sell) (optional)
        :returns The price
        """
        raise NotImplementedError

    def get_order_price_quantum(self, trading_pair: str, price: Decimal) -> Decimal:
        """Returns a price step, a minimum price increment for a given trading pair."""
        raise NotImplementedError

    def get_order_size_quantum(self, trading_pair: str, order_size: Decimal) -> Decimal:
        """Returns an order amount step, a minimum amount increment for a given trading pair."""
        raise NotImplementedError

    def quantize_order_price(self, trading_pair: str, price: Decimal) -> Decimal:
        """Applies trading rule to quantize order price."""
        return self.c_quantize_order_price(trading_pair, price)

    def quantize_order_amount(self, trading_pair: str, amount: Decimal) -> Decimal:
        """Applies trading rule to quantize order amount."""
        return self.c_quantize_order_amount(trading_pair, amount)

    async def get_quote_price(self, trading_pair: str, is_buy: bool, amount: Decimal) -> Decimal:
        """
        Returns a quote price (or exchange rate) for a given amount, like asking how much does it cost to buy 4 apples?
        :param trading_pair: The market trading pair
        :param is_buy: True for buy order, False for sell order
        :param amount: The order amount
        :return The quoted price
        """
        raise NotImplementedError

    async def get_order_price(self, trading_pair: str, is_buy: bool, amount: Decimal) -> Decimal:
        """
        Returns a price required for order submission, this price could differ from the quote price (e.g. for
        an exchange with order book).
        :param trading_pair: The market trading pair
        :param is_buy: True for buy order, False for sell order
        :param amount: The order amount
        :return The price to specify in an order.
        """
        raise NotImplementedError

    def add_trade_fills_from_market_recorder(self, current_trade_fills: Set[TradeFillOrderDetails]) -> None:
        """Gets updates from new records in TradeFill table. Used in is_confirmed_new_order_filled_event."""
        self._current_trade_fills.update(current_trade_fills)

    def add_exchange_order_ids_from_market_recorder(self, current_exchange_order_ids: Dict[str, str]) -> None:
        """Gets updates from new orders in Order table. Used in connector _history_reconciliation."""
        self._exchange_order_ids.update(current_exchange_order_ids)

    def is_confirmed_new_order_filled_event(
        self, exchange_trade_id: str, exchange_order_id: str, trading_pair: str
    ) -> bool:
        """
        Returns True if order to be filled is not already present in TradeFill entries.
        This is intended to avoid duplicated order fills in local DB.
        """
        return (
            TradeFillOrderDetails(self.display_name, exchange_trade_id, trading_pair) not in self._current_trade_fills
        ) and (exchange_order_id in set(self._exchange_order_ids.keys()))

    def trade_fee_schema(self):
        if self._trade_fee_schema is None:
            self._trade_fee_schema = TradeFeeSchemaLoader.configured_schema_for_exchange(exchange_name=self.name)
        return self._trade_fee_schema

    async def all_trading_pairs(self) -> List[str]:
        """
        List of all trading pairs supported by the connector.
        :return: List of trading pair symbols in the Hummingbot format
        """
        raise NotImplementedError

    async def _update_balances(self) -> None:
        """Update local balances requesting the latest information from the exchange."""
        raise NotImplementedError

    def _time(self) -> float:
        """
        Method created to enable tests to mock the machine time.
        :return: The machine time (time.time())
        """
        return time.time()

    async def _sleep(self, delay: float) -> None:
        """Method created to enable tests to prevent processes from sleeping."""
        await asyncio.sleep(delay)
