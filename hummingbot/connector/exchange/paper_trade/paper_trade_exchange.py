import asyncio
import math
import random
from collections import defaultdict, deque
from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from async_utils.core import safe_ensure_future

from hummingbot.connector.budget_checker import BudgetChecker
from hummingbot.connector.connector_metrics_collector import DummyMetricsCollector
from hummingbot.connector.exchange.paper_trade.trading_pair import TradingPair
from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.core.clock import Clock  # noqa: F401 – kept for runtime callers
from hummingbot.core.data_type.cancellation_result import CancellationResult
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.composite_order_book import CompositeOrderBook
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.order_book_tracker import OrderBookTracker
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.event_listener import EventListener
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    LimitOrderStatus,
    MarketEvent,
    MarketOrderFailureEvent,
    OrderBookEvent,
    OrderBookTradeEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.core.network_iterator import NetworkStatus
from hummingbot.core.utils.estimate_fee import build_trade_fee

if TYPE_CHECKING:
    pass

ptm_logger = None
s_decimal_0 = Decimal(0)


# ---------------------------------------------------------------------------
# Helper: sort key for limit orders – mirrors C++ operator< in LimitOrder.cpp
# Sorted ascending by (price, client_order_id) so that:
#   - Bids are iterated in reverse (highest price first)
#   - Asks are iterated forward (lowest price first)
# ---------------------------------------------------------------------------


def _limit_order_sort_key(order: LimitOrder) -> tuple:
    return (order.price, order.client_order_id)


def _insort_limit_order(order_list: list, order: LimitOrder) -> None:
    """Insert *order* into *order_list* maintaining ascending price order."""
    lo, hi = 0, len(order_list)
    key = _limit_order_sort_key(order)
    while lo < hi:
        mid = (lo + hi) // 2
        if _limit_order_sort_key(order_list[mid]) < key:
            lo = mid + 1
        else:
            hi = mid
    order_list.insert(lo, order)


def _remove_limit_order_by_id(order_list: list, client_order_id: str) -> bool:
    """Remove the first order matching *client_order_id*.  Returns True on success."""
    for i, o in enumerate(order_list):
        if o.client_order_id == client_order_id:
            del order_list[i]
            return True
    return False


# ---------------------------------------------------------------------------
# Tiny data classes (formerly cdef classes)
# ---------------------------------------------------------------------------


class QuantizationParams:
    def __init__(
        self,
        trading_pair: str,
        price_precision: int,
        price_decimals: int,
        order_size_precision: int,
        order_size_decimals: int,
    ):
        self.trading_pair = trading_pair
        self.price_precision = price_precision
        self.price_decimals = price_decimals
        self.order_size_precision = order_size_precision
        self.order_size_decimals = order_size_decimals

    def __repr__(self) -> str:
        return (
            f"QuantizationParams('{self.trading_pair}', {self.price_precision}, {self.price_decimals}, "
            f"{self.order_size_precision}, {self.order_size_decimals})"
        )


class QueuedOrder:
    def __init__(
        self,
        create_timestamp: float,
        order_id: str,
        is_buy: bool,
        trading_pair: str,
        amount: Decimal,
    ):
        self.create_timestamp = create_timestamp
        self._order_id = order_id
        self._is_buy = is_buy
        self._trading_pair = trading_pair
        self._amount = amount

    @property
    def timestamp(self) -> float:
        return self.create_timestamp

    @property
    def order_id(self) -> str:
        return self._order_id

    @property
    def is_buy(self) -> bool:
        return self._is_buy

    @property
    def trading_pair(self) -> str:
        return self._trading_pair

    @property
    def amount(self) -> Decimal:
        return self._amount

    def __repr__(self) -> str:
        return (
            f"QueuedOrder({self.create_timestamp}, '{self.order_id}', {self.is_buy}, '{self.trading_pair}', "
            f"{self.amount})"
        )


# ---------------------------------------------------------------------------
# Nested EventListener subclasses
# Uses def __call__ because the Cython PubSub dispatch path calls
# EventListener.c_call(arg) which in turn calls self(arg) == __call__.
# A plain "def c_call" on a Python subclass would NOT override the inherited
# cdef c_call at the C level, so __call__ is the correct protocol entry point.
# ---------------------------------------------------------------------------


class OrderBookTradeListener(EventListener):
    def __init__(self, market: ExchangeBase):
        super().__init__()
        self._market = market

    def __call__(self, event_object):
        try:
            self._market.match_trade_to_limit_orders(event_object)
        except Exception:
            self.logger().error("Error call trade listener.", exc_info=True)


class OrderBookMarketOrderFillListener(EventListener):
    def __init__(self, market: ExchangeBase):
        super().__init__()
        self._market = market

    def __call__(self, event_object):
        if event_object.trading_pair not in self._market.order_books or event_object.order_type != OrderType.MARKET:
            return
        order_book = self._market.order_books[event_object.trading_pair]
        order_book.record_filled_order(event_object)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PaperTradeExchange(ExchangeBase):
    TRADE_EXECUTION_DELAY = 5.0
    ORDER_FILLED_EVENT_TAG = MarketEvent.OrderFilled.value
    SELL_ORDER_COMPLETED_EVENT_TAG = MarketEvent.SellOrderCompleted.value
    BUY_ORDER_COMPLETED_EVENT_TAG = MarketEvent.BuyOrderCompleted.value
    MARKET_ORDER_CANCELED_EVENT_TAG = MarketEvent.OrderCancelled.value
    MARKET_ORDER_FAILURE_EVENT_TAG = MarketEvent.OrderFailure.value
    ORDER_BOOK_TRADE_EVENT_TAG = OrderBookEvent.TradeEvent.value
    MARKET_SELL_ORDER_CREATED_EVENT_TAG = MarketEvent.SellOrderCreated.value
    MARKET_BUY_ORDER_CREATED_EVENT_TAG = MarketEvent.BuyOrderCreated.value

    def __init__(
        self,
        order_book_tracker: OrderBookTracker,
        target_market: Callable,
        exchange_name: str,
        balance_asset_limit: Optional[Dict[str, Dict[str, Decimal]]] = None,
        rate_limits_share_pct: Decimal = Decimal("100"),
    ):
        # Pre-initialize attrs read by properties (e.g. display_name) that the
        # inherited ConnectorBase.__init__ chain may invoke before we return.
        self._exchange_name = exchange_name
        self._account_balances: Dict[str, Decimal] = {}
        self._account_available_balances: Dict[str, Decimal] = {}
        self._paper_trade_market_initialized: bool = False
        self._trading_pairs: Dict[str, TradingPair] = {}
        self._queued_orders: deque = deque()
        self._quantization_params: Dict[str, QuantizationParams] = {}
        # _bid_limit_orders / _ask_limit_orders: dict[trading_pair_str, list[LimitOrder]]
        # Lists are kept sorted ascending by (price, client_order_id) – mirrors the C++ cpp_set ordering.
        self._bid_limit_orders: Dict[str, List[LimitOrder]] = {}
        self._ask_limit_orders: Dict[str, List[LimitOrder]] = {}
        self._target_market = target_market
        self._order_book_trade_listener = OrderBookTradeListener(self)
        self._market_order_filled_listener = OrderBookMarketOrderFillListener(self)
        self._budget_checker = BudgetChecker(exchange=self)
        # Trade volume metrics should never be gathered for paper trade connector
        self._trade_volume_metric_collector = DummyMetricsCollector()

        order_book_tracker.data_source.order_book_create_function = lambda: CompositeOrderBook()
        super().__init__(balance_asset_limit, rate_limits_share_pct)
        # These require the inherited connector infrastructure to be ready:
        self._set_order_book_tracker(order_book_tracker)
        self.add_listener(MarketEvent.OrderFilled, self._market_order_filled_listener)

    @property
    def budget_checker(self) -> BudgetChecker:
        return self._budget_checker

    @classmethod
    def random_order_id(cls, order_side: str, trading_pair: str) -> str:
        vals = [random.choice(range(0, 256)) for i in range(0, 13)]
        return f"{order_side}://" + trading_pair + "/" + "".join([f"{val:02x}" for val in vals])

    def init_paper_trade_market(self):
        for trading_pair_str, order_book in self.order_book_tracker.order_books.items():
            assert type(order_book) is CompositeOrderBook
            base_asset, quote_asset = self.split_trading_pair(trading_pair_str)
            self._trading_pairs[self._target_market.convert_from_exchange_trading_pair(trading_pair_str)] = TradingPair(
                trading_pair_str, base_asset, quote_asset
            )
            order_book.add_listener(
                OrderBookEvent.TradeEvent,
                self._order_book_trade_listener,
            )

    def split_trading_pair(self, trading_pair: str) -> Tuple[str, str]:
        return self._target_market.split_trading_pair(trading_pair)

    #  <editor-fold desc="Property">
    @property
    def trading_pair(self) -> Dict[str, TradingPair]:
        return self._trading_pairs

    @property
    def trading_pairs(self) -> List[str]:
        return [trading_pair for trading_pair in self._trading_pairs]

    @property
    def name(self) -> str:
        return self._exchange_name

    @property
    def display_name(self) -> str:
        return f"{self._exchange_name}_PaperTrade"

    @property
    def order_books(self) -> Dict[str, CompositeOrderBook]:
        return self.order_book_tracker.order_books

    @property
    def status_dict(self) -> Dict[str, bool]:
        return {"order_books_initialized": self.order_book_tracker and len(self.order_book_tracker.order_books) > 0}

    @property
    def ready(self):
        if not self.order_book_tracker.ready:
            return False
        if all(self.status_dict.values()):
            if not self._paper_trade_market_initialized:
                self.init_paper_trade_market()
                self._paper_trade_market_initialized = True
            return True
        else:
            return False

    @property
    def queued_orders(self) -> List[QueuedOrder]:
        return self._queued_orders

    @property
    def limit_orders(self) -> List[LimitOrder]:
        retval = []

        # Bids: sorted ascending; iterate in reverse (highest bid first)
        for order_list in self._bid_limit_orders.values():
            for order in reversed(order_list):
                retval.append(order)

        # Asks: sorted ascending; iterate forward (lowest ask first)
        for order_list in self._ask_limit_orders.values():
            for order in order_list:
                retval.append(order)

        return retval

    @property
    def on_hold_balances(self) -> Dict[str, Decimal]:
        _on_hold_balances = defaultdict(Decimal)
        for limit_order in self.limit_orders:
            if limit_order.is_buy:
                _on_hold_balances[limit_order.quote_currency] += limit_order.quantity * limit_order.price
            else:
                _on_hold_balances[limit_order.base_currency] += limit_order.quantity
        return _on_hold_balances

    @property
    def available_balances(self) -> Dict[str, Decimal]:
        _available_balances = self._account_balances.copy()
        for trading_pair_str, balance in _available_balances.items():
            _available_balances[trading_pair_str] -= self.on_hold_balances[trading_pair_str]
        return _available_balances

    # </editor-fold>

    def start(self, clock: Clock, timestamp: float):
        super().start(clock, timestamp)

    async def start_network(self):
        await self.stop_network()
        self.order_book_tracker.start()

    async def stop_network(self):
        self.order_book_tracker.stop()

    async def check_network(self) -> NetworkStatus:
        return NetworkStatus.CONNECTED

    def c_set_balance(self, currency: str, balance: object):
        self._account_balances[currency.upper()] = Decimal(balance)

    def get_balance(self, currency: str) -> object:
        if currency.upper() not in self._account_balances:
            self.logger().warning(f"Account balance does not have asset {currency.upper()}.")
            return Decimal(0.0)
        return self._account_balances[currency.upper()]

    def tick(self, timestamp: float):
        super().tick(timestamp)
        self.c_process_market_orders()
        self.c_process_crossed_limit_orders()

    def buy(
        self,
        trading_pair_str: str,
        amount: object,
        order_type: object = OrderType.MARKET,
        price: object = s_decimal_0,
        **kwargs,
    ) -> str:
        if trading_pair_str not in self._trading_pairs:
            raise ValueError(f"Trading pair '{trading_pair_str}' does not existing in current data set.")

        order_id: str = self.random_order_id("buy", trading_pair_str)
        quote_asset: str = self._trading_pairs[trading_pair_str].quote_asset
        base_asset: str = self._trading_pairs[trading_pair_str].base_asset

        quantized_price = (
            self.quantize_order_price(trading_pair_str, price) if order_type is OrderType.LIMIT else s_decimal_0
        )
        quantized_amount = self.quantize_order_amount(trading_pair_str, amount)

        if order_type is OrderType.MARKET:
            self._queued_orders.append(
                QueuedOrder(self.current_timestamp, order_id, True, trading_pair_str, quantized_amount)
            )
        elif order_type is OrderType.LIMIT:
            if trading_pair_str not in self._bid_limit_orders:
                self._bid_limit_orders[trading_pair_str] = []
            new_order = LimitOrder(
                order_id,
                trading_pair_str,
                True,
                base_asset,
                quote_asset,
                quantized_price,
                quantized_amount,
                None,
                int(self.current_timestamp * 1e6),
                LimitOrderStatus.UNKNOWN,
            )
            _insort_limit_order(self._bid_limit_orders[trading_pair_str], new_order)

        safe_ensure_future(
            self.trigger_event_async(
                MarketEvent.BuyOrderCreated,
                BuyOrderCreatedEvent(
                    self.current_timestamp,
                    order_type,
                    trading_pair_str,
                    quantized_amount,
                    quantized_price,
                    order_id,
                    self.current_timestamp,
                ),
            )
        )
        return order_id

    def sell(
        self,
        trading_pair_str: str,
        amount: object,
        order_type: object = OrderType.MARKET,
        price: object = s_decimal_0,
        **kwargs,
    ) -> str:
        if trading_pair_str not in self._trading_pairs:
            raise ValueError(f"Trading pair '{trading_pair_str}' does not existing in current data set.")

        order_id: str = self.random_order_id("sell", trading_pair_str)
        base_asset: str = self._trading_pairs[trading_pair_str].base_asset
        quote_asset: str = self._trading_pairs[trading_pair_str].quote_asset

        quantized_price = (
            self.quantize_order_price(trading_pair_str, price) if order_type is OrderType.LIMIT else s_decimal_0
        )
        quantized_amount = self.quantize_order_amount(trading_pair_str, amount)

        if order_type is OrderType.MARKET:
            self._queued_orders.append(
                QueuedOrder(self.current_timestamp, order_id, False, trading_pair_str, quantized_amount)
            )
        elif order_type is OrderType.LIMIT:
            if trading_pair_str not in self._ask_limit_orders:
                self._ask_limit_orders[trading_pair_str] = []
            new_order = LimitOrder(
                order_id,
                trading_pair_str,
                False,
                base_asset,
                quote_asset,
                quantized_price,
                quantized_amount,
                None,
                int(self.current_timestamp * 1e6),
                LimitOrderStatus.UNKNOWN,
            )
            _insort_limit_order(self._ask_limit_orders[trading_pair_str], new_order)

        safe_ensure_future(
            self.trigger_event_async(
                MarketEvent.SellOrderCreated,
                SellOrderCreatedEvent(
                    self.current_timestamp,
                    order_type,
                    trading_pair_str,
                    quantized_amount,
                    quantized_price,
                    order_id,
                    self.current_timestamp,
                ),
            )
        )
        return order_id

    def c_execute_buy(self, order_id: str, trading_pair_str: str, amount: object):
        quote_asset: str = self._trading_pairs[trading_pair_str].quote_asset
        base_asset: str = self._trading_pairs[trading_pair_str].base_asset
        quote_balance: object = self.get_balance(quote_asset)
        base_balance: object = self.get_balance(base_asset)

        order_book = self.order_books[trading_pair_str]

        buy_entries = order_book.simulate_buy(float(amount))

        # Get the weighted average price of the trade
        avg_price = Decimal(0)
        for entry in buy_entries:
            avg_price += Decimal(entry.price) * Decimal(entry.amount)
        avg_price = avg_price / amount

        order_candidate = OrderCandidate(
            trading_pair=trading_pair_str,
            # Market orders are not maker orders
            is_maker=False,
            order_type=OrderType.MARKET,
            order_side=TradeType.BUY,
            amount=amount,
            price=avg_price,
            from_total_balances=True,
        )

        adjusted_order_candidate = self._budget_checker.populate_collateral_entries(order_candidate)

        # Quote currency used, including fees.
        paid_amount = adjusted_order_candidate.order_collateral.amount
        # Base currency acquired, including fees.
        acquired_amount = adjusted_order_candidate.potential_returns.amount

        # It's not possible to fulfill the order, the possible acquired amount is less than requested
        if paid_amount > quote_balance:
            self.logger().warning(
                f"Insufficient {quote_asset} balance available for buy order. "
                f"{quote_balance} {quote_asset} available vs. "
                f"{paid_amount} {quote_asset} required for the order."
            )
            self.trigger_event(
                MarketEvent.OrderFailure,
                MarketOrderFailureEvent(self.current_timestamp, order_id, OrderType.MARKET),
            )
            return

        # The order was successfully executed
        self.c_set_balance(quote_asset, quote_balance - paid_amount)
        self.c_set_balance(base_asset, base_balance + acquired_amount)

        # add fee
        fees = build_trade_fee(
            exchange=self.name,
            is_maker=False,
            base_currency="",
            quote_currency="",
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=Decimal("0"),
            price=Decimal("0"),
        )

        order_filled_events = OrderFilledEvent.order_filled_events_from_order_book_rows(
            self.current_timestamp, order_id, trading_pair_str, TradeType.BUY, OrderType.MARKET, fees, buy_entries
        )

        for order_filled_event in order_filled_events:
            self.trigger_event(MarketEvent.OrderFilled, order_filled_event)

        self.trigger_event(
            MarketEvent.BuyOrderCompleted,
            BuyOrderCompletedEvent(
                self.current_timestamp,
                order_id,
                base_asset,
                quote_asset,
                acquired_amount,
                paid_amount,
                OrderType.MARKET,
            ),
        )

    def c_execute_sell(self, order_id: str, trading_pair_str: str, amount: object):
        quote_asset: str = self._trading_pairs[trading_pair_str].quote_asset
        base_asset: str = self._trading_pairs[trading_pair_str].base_asset
        quote_balance: object = self.get_balance(quote_asset)
        base_balance: object = self.get_balance(base_asset)

        order_book = self.order_books[trading_pair_str]

        sell_entries = order_book.simulate_sell(float(amount))

        # Get the weighted average price of the trade
        avg_price = Decimal(0)
        for entry in sell_entries:
            avg_price += Decimal(entry.price) * Decimal(entry.amount)
        avg_price = avg_price / amount

        order_candidate = OrderCandidate(
            trading_pair=trading_pair_str,
            # Market orders are not maker orders
            is_maker=False,
            order_type=OrderType.MARKET,
            order_side=TradeType.SELL,
            amount=amount,
            price=avg_price,
            from_total_balances=True,
        )

        adjusted_order_candidate = self._budget_checker.populate_collateral_entries(order_candidate)

        # Base currency used, including fees.
        sold_amount = adjusted_order_candidate.order_collateral.amount
        # Quote currency acquired, including fees.
        acquired_amount = adjusted_order_candidate.potential_returns.amount

        # It's not possible to fulfill the order, the possible sold amount is less than requested
        if sold_amount > base_balance:
            self.logger().warning(
                f"Insufficient {base_asset} balance available for sell order. "
                f"{base_balance} {base_asset} available vs. "
                f"{amount} {base_asset} required for the order."
            )
            self.trigger_event(
                MarketEvent.OrderFailure,
                MarketOrderFailureEvent(self.current_timestamp, order_id, OrderType.MARKET),
            )
            return

        # The order was successfully executed
        self.c_set_balance(quote_asset, quote_balance + acquired_amount)
        self.c_set_balance(base_asset, base_balance - sold_amount)

        # add fee
        fees = build_trade_fee(
            exchange=self.name,
            is_maker=False,
            base_currency="",
            quote_currency="",
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=Decimal("0"),
            price=Decimal("0"),
        )

        order_filled_events = OrderFilledEvent.order_filled_events_from_order_book_rows(
            self.current_timestamp, order_id, trading_pair_str, TradeType.SELL, OrderType.MARKET, fees, sell_entries
        )

        for order_filled_event in order_filled_events:
            self.trigger_event(MarketEvent.OrderFilled, order_filled_event)

        self.trigger_event(
            MarketEvent.SellOrderCompleted,
            SellOrderCompletedEvent(
                self.current_timestamp,
                order_id,
                base_asset,
                quote_asset,
                sold_amount,
                acquired_amount,
                OrderType.MARKET,
            ),
        )

    def c_process_market_orders(self):
        while len(self._queued_orders) > 0:
            front_order: QueuedOrder = self._queued_orders[0]
            if front_order.create_timestamp <= self.current_timestamp - self.TRADE_EXECUTION_DELAY:
                self._queued_orders.popleft()
                try:
                    if front_order.is_buy:
                        self.c_execute_buy(front_order.order_id, front_order.trading_pair, front_order.amount)
                    else:
                        self.c_execute_sell(front_order.order_id, front_order.trading_pair, front_order.amount)
                except Exception:
                    self.logger().error("Error executing queued order.", exc_info=True)
            else:
                return

    def c_delete_limit_order(
        self,
        limit_orders_map: Dict[str, List[LimitOrder]],
        trading_pair_str: str,
        client_order_id: str,
    ) -> bool:
        """Remove a single limit order from the map by its client_order_id.

        Mirrors the C++ erase-iterator semantics: removes the entry and, if the
        per-trading-pair list becomes empty, removes the trading-pair key too.
        """
        try:
            order_list = limit_orders_map.get(trading_pair_str)
            if order_list is None:
                return False
            removed = _remove_limit_order_by_id(order_list, client_order_id)
            if removed and not order_list:
                del limit_orders_map[trading_pair_str]
            return removed
        except Exception:
            self.logger().error("Error deleting limit order.", exc_info=True)
            return False

    def c_process_limit_bid_order(
        self,
        limit_orders_map: Dict[str, List[LimitOrder]],
        trading_pair_str: str,
        order: LimitOrder,
    ):
        quote_asset: str = order.quote_currency
        base_asset: str = order.base_currency
        order_id: str = order.client_order_id
        amount: object = order.quantity
        price: object = order.price
        quote_balance: object = self.get_balance(quote_asset)
        base_balance: object = self.get_balance(base_asset)

        order_candidate = OrderCandidate(
            trading_pair=order.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=amount,
            price=price,
            from_total_balances=True,
        )

        adjusted_order_candidate = self._budget_checker.populate_collateral_entries(order_candidate)

        # Quote currency used, including fees.
        paid_amount = adjusted_order_candidate.order_collateral.amount
        # Base currency acquired, including fees.
        acquired_amount = adjusted_order_candidate.potential_returns.amount

        # It's not possible to fulfill the order, the possible acquired amount is less than requested
        if paid_amount > quote_balance:
            self.logger().warning(
                f"Not enough {quote_asset} balance to fill limit buy order on {order.trading_pair}. "
                f"{paid_amount:.8g} {quote_asset} needed vs. "
                f"{quote_balance:.8g} {quote_asset} available."
            )
            self.c_delete_limit_order(limit_orders_map, trading_pair_str, order_id)
            self.trigger_event(
                MarketEvent.OrderCancelled,
                OrderCancelledEvent(self.current_timestamp, order_id),
            )
            return

        # The order was successfully executed
        self.c_set_balance(quote_asset, quote_balance - paid_amount)
        self.c_set_balance(base_asset, base_balance + acquired_amount)

        # add fee
        fees = build_trade_fee(
            exchange=self.name,
            is_maker=True,
            base_currency="",
            quote_currency="",
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=Decimal("0"),
            price=Decimal("0"),
        )

        # Emit the trade and order completed events.
        self.trigger_event(
            MarketEvent.OrderFilled,
            OrderFilledEvent(
                self.current_timestamp,
                order_id,
                order.trading_pair,
                TradeType.BUY,
                OrderType.LIMIT,
                price,
                amount,
                fees,
                exchange_trade_id=str(int(self._time() * 1e6)),
            ),
        )

        self.trigger_event(
            MarketEvent.BuyOrderCompleted,
            BuyOrderCompletedEvent(
                self.current_timestamp,
                order_id,
                base_asset,
                quote_asset,
                acquired_amount,
                paid_amount,
                OrderType.LIMIT,
            ),
        )
        self.c_delete_limit_order(limit_orders_map, trading_pair_str, order_id)

    def c_process_limit_ask_order(
        self,
        limit_orders_map: Dict[str, List[LimitOrder]],
        trading_pair_str: str,
        order: LimitOrder,
    ):
        quote_asset: str = order.quote_currency
        base_asset: str = order.base_currency
        order_id: str = order.client_order_id
        amount: object = order.quantity
        price: object = order.price
        quote_balance: object = self.get_balance(quote_asset)
        base_balance: object = self.get_balance(base_asset)

        order_candidate = OrderCandidate(
            trading_pair=order.trading_pair,
            # Market orders are not maker orders
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=amount,
            price=price,
            from_total_balances=True,
        )

        adjusted_order_candidate = self._budget_checker.populate_collateral_entries(order_candidate)

        # Base currency used, including fees.
        sold_amount = adjusted_order_candidate.order_collateral.amount
        # Quote currency acquired, including fees.
        acquired_amount = adjusted_order_candidate.potential_returns.amount

        # It's not possible to fulfill the order, the possible sold amount is less than requested
        if sold_amount > base_balance:
            self.logger().warning(
                f"Not enough {base_asset} balance to fill limit sell order on {order.trading_pair}. "
                f"{sold_amount:.8g} {base_asset} needed vs. "
                f"{base_balance:.8g} {base_asset} available."
            )
            self.c_delete_limit_order(limit_orders_map, trading_pair_str, order_id)
            self.trigger_event(
                MarketEvent.OrderCancelled,
                OrderCancelledEvent(self.current_timestamp, order_id),
            )
            return

        # The order was successfully executed
        self.c_set_balance(quote_asset, quote_balance + acquired_amount)
        self.c_set_balance(base_asset, base_balance - sold_amount)

        # add fee
        fees = build_trade_fee(
            exchange=self.name,
            is_maker=True,
            base_currency="",
            quote_currency="",
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=Decimal("0"),
            price=Decimal("0"),
        )

        # Emit the trade and order completed events.
        self.trigger_event(
            MarketEvent.OrderFilled,
            OrderFilledEvent(
                self.current_timestamp,
                order_id,
                order.trading_pair,
                TradeType.SELL,
                OrderType.LIMIT,
                price,
                amount,
                fees,
                exchange_trade_id=str(int(self._time() * 1e6)),
            ),
        )

        self.trigger_event(
            MarketEvent.SellOrderCompleted,
            SellOrderCompletedEvent(
                self.current_timestamp,
                order_id,
                base_asset,
                quote_asset,
                sold_amount,
                acquired_amount,
                OrderType.LIMIT,
            ),
        )
        self.c_delete_limit_order(limit_orders_map, trading_pair_str, order_id)

    def c_process_limit_order(
        self,
        is_buy: bool,
        limit_orders_map: Dict[str, List[LimitOrder]],
        trading_pair_str: str,
        order: LimitOrder,
    ):
        try:
            if is_buy:
                self.c_process_limit_bid_order(limit_orders_map, trading_pair_str, order)
            else:
                self.c_process_limit_ask_order(limit_orders_map, trading_pair_str, order)
        except Exception:
            self.logger().error("Error processing limit order.", exc_info=True)

    def c_process_crossed_limit_orders_for_trading_pair(
        self,
        is_buy: bool,
        limit_orders_map: Dict[str, List[LimitOrder]],
        trading_pair_str: str,
    ):
        """
        Trigger limit orders when the opposite side of the order book has crossed the limit order's price.
        This implies someone was ready to fill the limit order, if that limit order was on the market.

        :param is_buy: are the limit orders on the bid side?
        :param limit_orders_map: the limit orders dict
        :param trading_pair_str: the trading pair being processed
        """
        order_list = limit_orders_map.get(trading_pair_str)
        if not order_list:
            return

        opposite_order_book_price = self.get_price(trading_pair_str, is_buy)

        # Collect orders to process (snapshot before mutation)
        orders_to_process: List[LimitOrder] = []

        if is_buy:
            # Iterate from highest bid (end of list) inward
            for order in reversed(order_list):
                if opposite_order_book_price > order.price:
                    break
                orders_to_process.append(order)
        else:
            # Iterate from lowest ask (start of list) inward
            for order in order_list:
                if opposite_order_book_price < order.price:
                    break
                orders_to_process.append(order)

        for order in orders_to_process:
            self.c_process_limit_order(is_buy, limit_orders_map, trading_pair_str, order)

    def c_process_crossed_limit_orders(self):
        for trading_pair_str in list(self._bid_limit_orders.keys()):
            self.c_process_crossed_limit_orders_for_trading_pair(True, self._bid_limit_orders, trading_pair_str)

        for trading_pair_str in list(self._ask_limit_orders.keys()):
            self.c_process_crossed_limit_orders_for_trading_pair(False, self._ask_limit_orders, trading_pair_str)

    # <editor-fold desc="Event listener functions">
    def c_match_trade_to_limit_orders(self, order_book_trade_event: object):
        """
        Trigger limit orders when incoming market orders have crossed the limit order's price.

        :param order_book_trade_event: trade event from order book
        """
        trading_pair: str = order_book_trade_event.trading_pair
        is_maker_buy: bool = order_book_trade_event.type is TradeType.SELL
        trade_price: object = order_book_trade_event.price

        limit_orders_map = self._bid_limit_orders if is_maker_buy else self._ask_limit_orders
        order_list = limit_orders_map.get(trading_pair)
        if not order_list:
            return

        # Collect orders to process (snapshot before mutation)
        orders_to_process: List[LimitOrder] = []

        if is_maker_buy:
            for order in reversed(order_list):
                if order.price <= trade_price:
                    break
                orders_to_process.append(order)
        else:
            for order in order_list:
                if order.price >= trade_price:
                    break
                orders_to_process.append(order)

        for order in orders_to_process:
            self.c_process_limit_order(is_maker_buy, limit_orders_map, trading_pair, order)

    # </editor-fold>

    def get_available_balance(self, currency: str) -> object:
        return self.available_balances.get(currency.upper(), s_decimal_0)

    async def cancel_all(self, timeout_seconds: float) -> List[CancellationResult]:
        cancellation_results = []
        for trading_pair_str in self._trading_pairs.keys():
            results = self.c_cancel_order_from_orders_map(self._bid_limit_orders, trading_pair_str, cancel_all=True)
            cancellation_results.extend(results)

        for trading_pair_str in self._trading_pairs.keys():
            results = self.c_cancel_order_from_orders_map(self._ask_limit_orders, trading_pair_str, cancel_all=True)
            cancellation_results.extend(results)
        return cancellation_results

    def c_cancel_order_from_orders_map(
        self,
        orders_map: Dict[str, List[LimitOrder]],
        trading_pair_str: str,
        cancel_all: bool = False,
        client_order_id: Optional[str] = None,
    ) -> object:
        cancellation_results = []
        try:
            order_list = orders_map.get(trading_pair_str)
            if not order_list:
                return []

            # Snapshot the orders to process so we can mutate during iteration
            if cancel_all:
                orders_to_cancel = list(order_list)
            else:
                orders_to_cancel = [o for o in order_list if o.client_order_id == client_order_id]

            for order in orders_to_cancel:
                cid = order.client_order_id
                delete_success = self.c_delete_limit_order(orders_map, trading_pair_str, cid)
                cancellation_results.append(CancellationResult(cid, delete_success))
                self.trigger_event(
                    MarketEvent.OrderCancelled,
                    OrderCancelledEvent(self.current_timestamp, cid),
                )
            return cancellation_results
        except Exception:
            self.logger().error("Error canceling order.", exc_info=True)

    def cancel(self, trading_pair_str: str, client_order_id: str):
        trade_type: str = client_order_id.split("://")[0]
        is_maker_buy: bool = trade_type.upper() == "BUY"
        limit_orders_map = self._bid_limit_orders if is_maker_buy else self._ask_limit_orders
        self.c_cancel_order_from_orders_map(limit_orders_map, trading_pair_str, False, client_order_id)

    def c_get_fee(
        self,
        base_asset: str,
        quote_asset: str,
        order_type: object,
        order_side: object,
        amount: object,
        price: object,
        is_maker: object = None,
    ) -> object:
        return build_trade_fee(
            self.name,
            is_maker=is_maker if is_maker is not None else order_type in [OrderType.LIMIT, OrderType.LIMIT_MAKER],
            base_currency=base_asset,
            quote_currency=quote_asset,
            order_type=order_type,
            order_side=order_side,
            amount=amount,
            price=price,
        )

    def c_get_order_book(self, trading_pair: str) -> OrderBook:
        if trading_pair not in self._trading_pairs:
            raise ValueError(f"No order book exists for '{trading_pair}'.")
        trading_pair = self._target_market.convert_to_exchange_trading_pair(trading_pair)
        return self._order_book_tracker.order_books[trading_pair]

    def get_price(self, trading_pair: str, is_buy: bool) -> Decimal:
        """Return top bid/ask price for a trading pair, quantized."""
        order_book: OrderBook = self.c_get_order_book(trading_pair)
        try:
            top_price = Decimal(str(order_book.get_price(is_buy)))
        except EnvironmentError:
            self.logger().warning(f"{'Ask' if is_buy else 'Bid'} orderbook for {trading_pair} is empty.")
            return Decimal("nan")
        return self.quantize_order_price(trading_pair, top_price)

    def get_order_price_quantum(self, trading_pair: str, price: object) -> object:
        if trading_pair in self._quantization_params:
            q_params: QuantizationParams = self._quantization_params[trading_pair]
            decimals_quantum = Decimal(f"1e-{q_params.price_decimals}")
            if price.is_finite() and price > s_decimal_0:
                precision_quantum = Decimal(f"1e{math.ceil(math.log10(price)) - q_params.price_precision}")
            else:
                precision_quantum = Decimal(0)
            return max(precision_quantum, decimals_quantum)
        else:
            return Decimal("1e-10")

    def get_order_size_quantum(self, trading_pair: str, order_size: object) -> object:
        if trading_pair in self._quantization_params:
            q_params: QuantizationParams = self._quantization_params[trading_pair]
            decimals_quantum = Decimal(f"1e-{q_params.order_size_decimals}")
            if order_size.is_finite() and order_size > s_decimal_0:
                precision_quantum = Decimal(f"1e{math.ceil(math.log10(order_size)) - q_params.order_size_precision}")
            else:
                precision_quantum = Decimal(0)
            return max(precision_quantum, decimals_quantum)
        else:
            return Decimal("1e-7")

    def quantize_order_price(self, trading_pair: str, price: object) -> object:
        price = Decimal("%.7g" % price)  # hard code to round to 8 significant digits
        price_quantum = self.get_order_price_quantum(trading_pair, price)
        return (price // price_quantum) * price_quantum

    def quantize_order_amount(self, trading_pair: str, amount: object, price: object = s_decimal_0) -> object:
        amount = amount.quantize(Decimal("1e-7"), rounding=ROUND_DOWN)
        if amount <= 1e-7:
            amount = Decimal("0")
        order_size_quantum = self.get_order_size_quantum(trading_pair, amount)
        return (amount // order_size_quantum) * order_size_quantum

    def get_all_balances(self) -> Dict[str, Decimal]:
        return self._account_balances.copy()

    # <editor-fold desc="Python wrapper for cdef functions">
    def match_trade_to_limit_orders(self, event_object: OrderBookTradeEvent):
        self.c_match_trade_to_limit_orders(event_object)

    def set_balance(self, currency: str, balance: Decimal):
        self.c_set_balance(currency, balance)

    # </editor-fold>

    def get_fee(
        self,
        base_currency: str,
        quote_currency: str,
        order_type: OrderType,
        order_side: TradeType,
        amount: Decimal,
        price: Decimal = s_decimal_0,
        is_maker: Optional[bool] = None,
    ):
        return self.c_get_fee(base_currency, quote_currency, order_type, order_side, amount, price, is_maker)

    def get_order_book(self, trading_pair: str) -> OrderBook:
        return self.c_get_order_book(trading_pair)

    def get_maker_order_type(self):
        return OrderType.LIMIT

    def get_taker_order_type(self):
        return OrderType.LIMIT

    async def trigger_event_async(self, event_tag: MarketEvent, event: object):
        await asyncio.sleep(0.01)
        self.trigger_event(event_tag, event)
