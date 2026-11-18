import asyncio
import contextlib
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional

from event_bus import EventBus

from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.connector.market_connector.order_book_adapter import OrderBookCache
from hummingbot.connector.market_connector.rules_converter import to_parent_trading_rule
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
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


class MarketConnectorExchange(ExchangeBase):
    """
    Generic parent ConnectorBase over any market-connector exchange core.

    The constructor performs no I/O and never calls ``descriptor.create``; the core is built and started in
    ``start_network``. Order entry (buy/sell/cancel) is not implemented yet.
    """

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
        self._event_bus: Optional[EventBus] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._book_cache = OrderBookCache()
        self._rules_cache: Dict[str, Any] = {}
        self._fee_schema = TradeFeeSchema(
            maker_percent_fee_decimal=descriptor.default_maker_fee,
            taker_percent_fee_decimal=descriptor.default_taker_fee,
        )

    # --- lifecycle -------------------------------------------------------------------------------------------

    async def start_network(self) -> None:
        if self._handle is not None:
            return
        bus = EventBus(name=self._connector_name)
        handle = await self._descriptor.create(self._config, event_bus=bus, **self._core_options)
        try:
            await handle.start()
            handle.core.track_pairs(self._trading_pairs)
        except BaseException:
            await handle.close()
            raise
        self._event_bus = bus
        self._handle = handle
        self._refresh_task = asyncio.ensure_future(self._refresh_order_books_loop())

    async def stop_network(self) -> None:
        task, handle = self._refresh_task, self._handle
        self._refresh_task = None
        self._handle = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        try:
            if handle is not None:
                await handle.close()
        finally:
            self._event_bus = None
            self._book_cache.clear()
            self._rules_cache.clear()

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
        return {}

    @property
    def limit_orders(self) -> List[Any]:
        return []

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
        raise NotImplementedError("P2")

    def sell(
        self, trading_pair: str, amount: Decimal, order_type=OrderType.MARKET, price: Decimal = s_decimal_NaN, **kwargs
    ) -> str:
        raise NotImplementedError("P2")

    def cancel(self, trading_pair: str, client_order_id: str):
        raise NotImplementedError("P2")
