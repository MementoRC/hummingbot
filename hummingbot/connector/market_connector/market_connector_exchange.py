import asyncio
import contextlib
from decimal import Decimal
from enum import Enum
import functools
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from data_type_primitives.trigger_spec import TriggerKind, TriggerSpec
from event_bus import EventBus

from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.connector.market_connector.events_converter import (
    EVENT_CLASSES,
    MARKET_EVENT_BY_ID,
    TerminalEventDeduper,
    to_parent_event,
    to_parent_order_type,
    to_parent_position_action,
)
from hummingbot.connector.market_connector.order_book_adapter import OrderBookCache
from hummingbot.connector.market_connector.order_tracker_shim import CoreOrderTracker
from hummingbot.connector.market_connector.rules_converter import to_parent_trading_rule
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.trade_fee import TradeFeeBase, TradeFeeSchema
from hummingbot.core.network_iterator import NetworkStatus

s_decimal_NaN = Decimal("nan")
s_decimal_0 = Decimal(0)

# Keys of a core's status_dict (KrakenConnectorCore with own_trades_ws); used only before the core exists.
PRE_START_STATUS_KEYS = (
    "symbols_mapping_initialized",
    "trading_rule_initialized",
    "account_balance",
    "status_polling_initialized",
    "user_stream_initialized",
)
DEFAULT_ORDER_TYPES = [OrderType.LIMIT, OrderType.MARKET, OrderType.LIMIT_MAKER]
# Optional order extras forwarded to the core when the caller supplies them via **kwargs.
CORE_ORDER_KWARGS = ("limit_price", "price_in_percent", "position_action", "expiration_ts", "trigger")
# Order types expressed through the core's ``trigger=TriggerSpec`` form.
TRIGGER_ORDER_TYPES = (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT)


class MarketConnectorExchange(ExchangeBase):
    """
    Generic parent ConnectorBase over any market-connector exchange core.

    The constructor performs no I/O and never calls ``descriptor.create``; the core is built and started in
    ``start_network``. Orders are delegated to the core; its bus events are translated to parent events.
    """

    LINKED_TICK_INTERVAL_S = 1.0

    def __init__(
        self,
        descriptor: Any,
        config: Mapping[str, Any],
        connector_name: str,
        trading_pairs: Optional[List[str]] = None,
        trading_required: bool = True,
        balance_asset_limit: Optional[Dict[str, Dict[str, Decimal]]] = None,
        rate_limits_share_pct: Decimal = Decimal("100"),
        core_options: Optional[Mapping[str, Any]] = None,
        order_book_refresh_interval: float = 5.0,
    ):
        # ConnectorBase.__init__ reads display_name, so the name must exist first.
        self._connector_name = connector_name
        super().__init__(balance_asset_limit, rate_limits_share_pct)
        self._descriptor = descriptor
        self._config = dict(config)
        self._trading_pairs: List[str] = list(trading_pairs or [])
        self._trading_required = trading_required
        self._core_options: Dict[str, Any] = dict(core_options or {})
        self._order_book_refresh_interval = order_book_refresh_interval
        self._handle: Any = None
        self._lifecycle_lock_obj: Optional[asyncio.Lock] = None
        self._event_bus: Optional[EventBus] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._linked_manager: Optional[Any] = None
        self._leg_fill_unsub: Optional[Callable[[], None]] = None
        self._linked_tick_task: Optional[asyncio.Task] = None
        self._event_subscriptions: List[Any] = []
        self._terminal_deduper = TerminalEventDeduper()
        self._order_tracker = CoreOrderTracker(lambda: self._core)
        self._book_cache = OrderBookCache()
        self._rules_cache: Dict[str, Any] = {}
        self._fee_schema = TradeFeeSchema(
            maker_percent_fee_decimal=descriptor.default_maker_fee,
            taker_percent_fee_decimal=descriptor.default_taker_fee,
        )

    # --- lifecycle -------------------------------------------------------------------------------------------

    @property
    def _lifecycle_lock(self) -> asyncio.Lock:
        # Created lazily so __init__ needs no running loop.
        if self._lifecycle_lock_obj is None:
            self._lifecycle_lock_obj = asyncio.Lock()
        return self._lifecycle_lock_obj

    async def start_network(self) -> None:
        async with self._lifecycle_lock:
            if self._handle is not None:
                return
            bus = EventBus(name=self._connector_name)
            handle = await self._descriptor.create(self._config, event_bus=bus, **self._core_options)
            try:
                await handle.start()
                self._subscribe_core_events(bus)
                handle.core.track_pairs(self._trading_pairs)
                await self._start_linked_manager(handle)
            except BaseException:
                self._release_linked_resources()
                self._unsubscribe_core_events()
                await handle.close()
                raise
            self._event_bus = bus
            self._handle = handle
            self._refresh_task = asyncio.ensure_future(self._refresh_order_books_loop())

    async def stop_network(self) -> None:
        async with self._lifecycle_lock:
            task, handle = self._refresh_task, self._handle
            self._refresh_task = None
            self._handle = None
            try:
                await self._stop_linked_manager()
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                if handle is not None:
                    # Close first: the core emits terminal events while draining, and those must still be delivered.
                    await handle.close()
            finally:
                self._unsubscribe_core_events()
                self._event_bus = None
                self._book_cache.clear()
                self._rules_cache.clear()

    # --- linked-order emulation (P4a) ------------------------------------------------------------------------

    @property
    def linked_order_manager(self) -> Optional[Any]:
        return self._linked_manager

    def _build_linked_venue(self, handle: Any) -> Optional[Tuple[Any, Any]]:
        """Return a ``(venue, codec)`` pair to enable emulated linked orders, or None (default)."""
        return None

    async def _start_linked_manager(self, handle: Any) -> None:
        pair = self._build_linked_venue(handle)
        if pair is None:
            return
        # Imported here so connectors without linked-order support never need strategy_framework.
        from strategy_framework.linked_orders import (
            EmulatedLinkedOrderManager,
            InMemoryLinkedGroupStore,
            LinkedOrderManagerConfig,
        )

        venue, codec = pair
        # cancel_orphans must stay False: on a shared account another host's live legs look like orphans.
        config = LinkedOrderManagerConfig(
            enabled=True,
            require_recover_before_place=True,
            grace_period_s=30,
            cancel_orphans=False,
            allow_volatile_store=True,
        )
        manager = EmulatedLinkedOrderManager(venue, codec, config, store=InMemoryLinkedGroupStore())
        self._linked_manager = manager
        # Order matters: subscribe, then recover, then tick.
        self._leg_fill_unsub = venue.subscribe(manager.on_fill)
        await manager.recover(list(self._trading_pairs))
        self._linked_tick_task = asyncio.ensure_future(self._linked_tick_loop())

    async def _linked_tick_loop(self) -> None:
        while True:
            try:
                await self._linked_manager.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger().exception("Linked order manager tick failed.")
            await asyncio.sleep(self.LINKED_TICK_INTERVAL_S)

    def _release_linked_resources(self) -> None:
        """Synchronous cleanup for a failed start: stop the tick task and unsubscribe, keep nothing."""
        task, unsub = self._linked_tick_task, self._leg_fill_unsub
        self._linked_tick_task = self._leg_fill_unsub = self._linked_manager = None
        if task is not None:
            task.cancel()
        if unsub is not None:
            unsub()

    async def _stop_linked_manager(self) -> None:
        task, unsub, manager = self._linked_tick_task, self._leg_fill_unsub, self._linked_manager
        try:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if unsub is not None:
                unsub()
            if manager is not None:
                await manager.shutdown()
        finally:
            self._linked_tick_task = self._leg_fill_unsub = self._linked_manager = None

    async def check_network(self) -> NetworkStatus:
        return NetworkStatus.CONNECTED if self._handle is not None else NetworkStatus.NOT_CONNECTED

    async def _refresh_order_books_loop(self) -> None:
        while True:
            for pair in list(self._trading_pairs):
                try:
                    await self._core.update_order_book(pair)
                except Exception:
                    self.logger().exception(f"Order book refresh failed for {pair}.")
            await asyncio.sleep(self._order_book_refresh_interval)

    @property
    def _core(self) -> Any:
        return None if self._handle is None else self._handle.core

    # --- status ----------------------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._connector_name

    @property
    def status_dict(self) -> Dict[str, bool]:
        core = self._core
        if core is None:
            return {key: False for key in PRE_START_STATUS_KEYS}
        return dict(core.status_dict)

    @property
    def ready(self) -> bool:
        core = self._core
        return core is not None and bool(core.ready)

    @property
    def trading_pairs(self) -> List[str]:
        return self._trading_pairs

    @property
    def in_flight_orders(self) -> Dict[str, Any]:
        return self._order_tracker.active_orders

    @property
    def limit_orders(self) -> List[Any]:
        return [order.to_limit_order() for order in self._order_tracker.active_orders.values()]

    def tick(self, timestamp: float) -> None:
        pass

    # --- market data -----------------------------------------------------------------------------------------

    def get_order_book(self, trading_pair: str) -> OrderBook:
        core = self._core
        snapshot = None if core is None else core.get_order_book(trading_pair)
        return self._book_cache.sync(trading_pair, snapshot)

    @property
    def order_books(self) -> Dict[str, OrderBook]:
        return {pair: self.get_order_book(pair) for pair in self._trading_pairs}

    def get_price(self, trading_pair: str, is_buy: bool, amount: Decimal = s_decimal_NaN) -> Decimal:
        core = self._core
        snapshot = None if core is None else core.get_order_book(trading_pair)
        if snapshot is None:
            return s_decimal_NaN
        entries = snapshot.asks if is_buy else snapshot.bids
        return entries[0].price if entries else s_decimal_NaN

    def get_price_by_type(self, trading_pair: str, price_type: PriceType) -> Decimal:
        core = self._core
        if core is None:
            return s_decimal_NaN
        price = core.get_price_by_type(trading_pair, price_type)
        if not price.is_nan() or price_type not in (PriceType.BestBid, PriceType.BestAsk, PriceType.MidPrice):
            return price
        bid, ask = self.get_price(trading_pair, False), self.get_price(trading_pair, True)
        if price_type is PriceType.BestBid:
            return bid
        if price_type is PriceType.BestAsk:
            return ask
        return (bid + ask) / 2

    # --- balances --------------------------------------------------------------------------------------------

    def get_balance(self, currency: str) -> Decimal:
        core = self._core
        return s_decimal_0 if core is None else core.get_balance(currency)

    def get_available_balance(self, currency: str) -> Decimal:
        core = self._core
        return s_decimal_0 if core is None else core.get_available_balance(currency)

    def _tracked_assets(self) -> List[str]:
        # The core exposes per-asset getters only, so aggregate over the assets of the tracked pairs.
        assets: List[str] = []
        for pair in self._trading_pairs:
            for asset in pair.split("-"):
                if asset not in assets:
                    assets.append(asset)
        return assets

    def get_all_balances(self) -> Dict[str, Decimal]:
        return {asset: self.get_balance(asset) for asset in self._tracked_assets()}

    @property
    def available_balances(self) -> Dict[str, Decimal]:
        return {asset: self.get_available_balance(asset) for asset in self._tracked_assets()}

    # --- trading rules and quantization ----------------------------------------------------------------------

    @property
    def trading_rules(self) -> Dict[str, TradingRule]:
        core = self._core
        if core is None:
            return {}
        converted: Dict[str, TradingRule] = {}
        for pair, rules in core.trading_rules.items():
            cached = self._rules_cache.get(pair)
            if cached is None or cached[0] is not rules:
                cached = (rules, to_parent_trading_rule(rules))
                self._rules_cache[pair] = cached
            converted[pair] = cached[1]
        return converted

    def quantize_order_amount(self, trading_pair: str, amount: Decimal) -> Decimal:
        core = self._core
        return amount if core is None else core.quantize_order_amount(trading_pair, amount)

    def quantize_order_price(self, trading_pair: str, price: Decimal) -> Decimal:
        core = self._core
        return price if core is None else core.quantize_order_price(trading_pair, price)

    def get_order_price_quantum(self, trading_pair: str, price: Decimal) -> Decimal:
        core = self._core
        return s_decimal_0 if core is None else core.get_order_price_quantum(trading_pair, price)

    def get_order_size_quantum(self, trading_pair: str, order_size: Decimal) -> Decimal:
        core = self._core
        return s_decimal_0 if core is None else core.get_order_size_quantum(trading_pair, order_size)

    # --- fees and capabilities -------------------------------------------------------------------------------

    def get_fee(
        self,
        base_currency: str,
        quote_currency: str,
        order_type: OrderType,
        order_side: TradeType,
        amount: Decimal,
        price: Decimal = s_decimal_NaN,
        is_maker: Optional[bool] = None,
    ) -> TradeFeeBase:
        if is_maker is None:
            is_maker = order_type is OrderType.LIMIT_MAKER
        percent = self._fee_schema.maker_percent_fee_decimal if is_maker else self._fee_schema.taker_percent_fee_decimal
        return TradeFeeBase.new_spot_fee(fee_schema=self._fee_schema, trade_type=order_side, percent=percent)

    def supported_order_types(self) -> List[OrderType]:
        declared = self._descriptor.order_capabilities.order_types
        types = [order_type for order_type in OrderType if order_type in declared]
        return types or list(DEFAULT_ORDER_TYPES)

    # --- order entry (P2) ------------------------------------------------------------------------------------

    def buy(
        self, trading_pair: str, amount: Decimal, order_type=OrderType.MARKET, price: Decimal = s_decimal_NaN, **kwargs
    ) -> str:
        return self._create_order(TradeType.BUY, trading_pair, amount, order_type, price, kwargs)

    def sell(
        self, trading_pair: str, amount: Decimal, order_type=OrderType.MARKET, price: Decimal = s_decimal_NaN, **kwargs
    ) -> str:
        return self._create_order(TradeType.SELL, trading_pair, amount, order_type, price, kwargs)

    def cancel(self, trading_pair: str, client_order_id: str) -> str:
        core = self._require_core()
        return core.cancel(trading_pair, client_order_id)

    def _require_core(self) -> Any:
        core = self._core
        if core is None:
            raise RuntimeError(f"{self._connector_name} is not started; call start_network first.")
        return core

    def _create_order(
        self,
        trade_type: TradeType,
        trading_pair: str,
        amount: Decimal,
        order_type: OrderType,
        price: Decimal,
        kwargs: Mapping[str, Any],
    ) -> str:
        core = self._require_core()
        if order_type not in self.supported_order_types():
            raise ValueError(f"{self._connector_name} does not support {order_type.name} orders.")
        # Only the core's documented extras pass through; other keys (e.g. executor bookkeeping) are dropped.
        extras = {key: kwargs[key] for key in CORE_ORDER_KWARGS if kwargs.get(key) is not None}
        ignored = sorted(key for key in kwargs if key not in CORE_ORDER_KWARGS)
        if ignored:
            self.logger().debug(f"{self._connector_name}: ignoring unsupported order kwargs {ignored}.")
        if "position_action" in extras:
            action = extras["position_action"]
            if not isinstance(action, Enum) or action.name not in PositionAction.__members__:
                raise TypeError(f"position_action must be a PositionAction member, got {action!r}.")
            extras["position_action"] = to_parent_position_action(action)
        method = core.buy if trade_type is TradeType.BUY else core.sell
        # *_LIMIT -> #85 P5 (limit-price source); trailing -> follow-up. Both keep the legacy call.
        if order_type in TRIGGER_ORDER_TYPES:
            if "trigger" not in extras:
                extras["trigger"] = self._build_trigger(price, bool(extras.pop("price_in_percent", False)))
            else:
                extras.pop("price_in_percent", None)
            return method(trading_pair, amount, to_parent_order_type(order_type), **extras)
        return method(
            trading_pair,
            amount,
            to_parent_order_type(order_type),
            price,
            **extras,
        )

    @staticmethod
    def _build_trigger(price: Decimal, in_percent: bool) -> TriggerSpec:
        value = Decimal(str(price))
        if in_percent:
            # Legacy price_in_percent is in percent points; TriggerSpec PERCENT takes a fraction.
            return TriggerSpec(kind=TriggerKind.PERCENT, value=value / 100)
        return TriggerSpec(kind=TriggerKind.ABSOLUTE, value=value)

    # --- core event bridge -----------------------------------------------------------------------------------

    def _subscribe_core_events(self, bus: EventBus) -> None:
        for event_id in EVENT_CLASSES:
            self._event_subscriptions.append(bus.subscribe(event_id, functools.partial(self._on_core_event, event_id)))

    def _unsubscribe_core_events(self) -> None:
        subscriptions, self._event_subscriptions = self._event_subscriptions, []
        for subscription in subscriptions:
            subscription.bus.unsubscribe(subscription)
        self._terminal_deduper.clear()

    def _on_core_event(self, event_id: int, mc_event: Any) -> None:
        try:
            if self._terminal_deduper.is_duplicate(event_id, getattr(mc_event, "order_id", None)):
                return
            self.trigger_event(MARKET_EVENT_BY_ID[event_id], to_parent_event(event_id, mc_event))
        except Exception:
            self.logger().exception(f"Failed to forward core event {event_id}.")
