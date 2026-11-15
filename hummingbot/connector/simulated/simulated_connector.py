"""ExchangePyBase drop-in over hb-market-simulator's hummingbot-free SimulatedConnectorCore.

The core owns the simulated exchange and the client-order-id <-> simulator-order-id
bookkeeping, and pushes OrderUpdate/TradeUpdate objects to registered listeners. This
shim feeds those into ClientOrderTracker and exposes the usual ExchangePyBase surface.

Type conversion: the sub-package emits hb-data-type-primitives (dtp) classes, while the
parent's OrderUpdate/TradeUpdate/OrderState/TradeFee are separate classes defined in
hummingbot.core.data_type. Every update crossing the boundary is converted here.
"""

import asyncio
from decimal import Decimal
from typing import Any

from data_type_primitives import OrderUpdate as DtpOrderUpdate, TradeUpdate as DtpTradeUpdate
from market_simulator.connector import SimulatedConnectorCore

from hummingbot.connector.constants import s_decimal_NaN
from hummingbot.connector.exchange_py_base import ExchangePyBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.api_throttler.data_types import RateLimit
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState, OrderUpdate, TradeUpdate
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.order_book_tracker_data_source import OrderBookTrackerDataSource
from hummingbot.core.data_type.trade_fee import TokenAmount, TradeFeeBase
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.web_assistant.auth import AuthBase
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory


def _to_hb_fee(fee: Any) -> TradeFeeBase:
    fee_cls = TradeFeeBase.fee_class_for_type(fee.type_descriptor_for_json())
    return fee_cls(
        percent=fee.percent,
        percent_token=fee.percent_token,
        flat_fees=[TokenAmount(token=f.token, amount=f.amount) for f in fee.flat_fees],
    )


def _to_hb_order_update(update: DtpOrderUpdate) -> OrderUpdate:
    return OrderUpdate(
        trading_pair=update.trading_pair,
        update_timestamp=update.update_timestamp,
        new_state=OrderState[update.new_state.name],
        client_order_id=update.client_order_id,
        exchange_order_id=update.exchange_order_id,
        misc_updates=update.misc_updates,
    )


def _to_hb_trade_update(update: DtpTradeUpdate) -> TradeUpdate:
    return TradeUpdate(
        trade_id=update.trade_id,
        client_order_id=update.client_order_id,
        exchange_order_id=update.exchange_order_id,
        trading_pair=update.trading_pair,
        fill_timestamp=update.fill_timestamp,
        fill_price=update.fill_price,
        fill_base_amount=update.fill_base_amount,
        fill_quote_amount=update.fill_quote_amount,
        fee=_to_hb_fee(update.fee),
        is_taker=update.is_taker,
    )


async def _park_forever() -> None:
    await asyncio.Event().wait()


class _NoOpOrderBookDataSource(OrderBookTrackerDataSource):
    """No-op order book source: there is no network, so streams just park.

    Order books stay empty; the simulator owns its own book inside the core.
    """

    async def get_last_traded_prices(self, trading_pairs: list[str], domain: str | None = None) -> dict[str, float]:
        return {}

    async def get_new_order_book(self, trading_pair: str) -> OrderBook:
        return self.order_book_create_function()

    async def listen_for_subscriptions(self):
        await _park_forever()

    async def listen_for_order_book_diffs(self, ev_loop: asyncio.AbstractEventLoop, output: asyncio.Queue):
        await _park_forever()

    async def listen_for_order_book_snapshots(self, ev_loop: asyncio.AbstractEventLoop, output: asyncio.Queue):
        await _park_forever()

    async def listen_for_trades(self, ev_loop: asyncio.AbstractEventLoop, output: asyncio.Queue):
        await _park_forever()

    async def subscribe_to_trading_pair(self, trading_pair: str) -> bool:
        return True

    async def unsubscribe_from_trading_pair(self, trading_pair: str) -> bool:
        return True


class _NoOpUserStreamDataSource(UserStreamTrackerDataSource):
    """No-op user stream: fills arrive through core listeners, not a stream."""

    @property
    def last_recv_time(self) -> float:
        return self._time()

    async def listen_for_user_stream(self, output: asyncio.Queue):
        await _park_forever()


class SimulatedConnector(ExchangePyBase):
    """hummingbot connector backed by a SimulatedConnectorCore (no network I/O)."""

    def __init__(
        self,
        core: SimulatedConnectorCore | None = None,
        balance_asset_limit: dict[str, dict[str, Decimal]] | None = None,
    ):
        # The base __init__ reads trading_pairs / rate_limits_rules, so the core must exist first.
        self._core = core or SimulatedConnectorCore()
        # Client ids currently inside _place_order, and order updates held back for them.
        self._placing_ids: set[str] = set()
        self._held_order_updates: dict[str, list[OrderUpdate]] = {}
        super().__init__(balance_asset_limit=balance_asset_limit)
        self._core.add_order_update_listener(self._on_core_order_update)
        self._core.add_trade_update_listener(self._on_core_trade_update)

    @property
    def core(self) -> SimulatedConnectorCore:
        return self._core

    # --- Listener bridge ---

    def _on_core_order_update(self, update: DtpOrderUpdate) -> None:
        hb_update = _to_hb_order_update(update)
        held = self._held_order_updates.get(hb_update.client_order_id)
        if held is not None:
            held.append(hb_update)
        else:
            self._order_tracker.process_order_update(hb_update)

    def _on_core_trade_update(self, update: DtpTradeUpdate) -> None:
        self._order_tracker.process_trade_update(_to_hb_trade_update(update))

    async def _place_order_and_process_update(self, order: InFlightOrder, **kwargs) -> str:
        """Hold core order updates emitted during placement until the base class has applied OPEN.

        The core emits synchronously inside place_order (instant MARKET fills, insufficient
        balance FAILED). The order is already tracked by then, but the base class schedules an
        OPEN update after _place_order returns; if a terminal update were processed first, that
        late OPEN would regress the order state. Trade updates are applied immediately.
        """
        client_order_id = order.client_order_id
        self._held_order_updates[client_order_id] = []
        try:
            exchange_order_id = await super()._place_order_and_process_update(order, **kwargs)
        except BaseException:
            self._held_order_updates.pop(client_order_id, None)
            raise
        for update in self._held_order_updates.pop(client_order_id, []):
            self._order_tracker.process_order_update(update)
        return exchange_order_id

    # --- Identity / configuration ---

    @property
    def name(self) -> str:
        return self._core.name

    @property
    def authenticator(self) -> AuthBase | None:
        return None

    @property
    def rate_limits_rules(self) -> list[RateLimit]:
        return []

    @property
    def domain(self) -> str:
        return ""

    @property
    def client_order_id_max_length(self) -> int:
        return 32

    @property
    def client_order_id_prefix(self) -> str:
        return "SIM"

    @property
    def trading_rules_request_path(self) -> str:
        return ""

    @property
    def trading_pairs_request_path(self) -> str:
        return ""

    @property
    def check_network_request_path(self) -> str:
        return ""

    @property
    def trading_pairs(self) -> list[str]:
        return self._core.trading_pairs

    @property
    def is_cancel_request_in_exchange_synchronous(self) -> bool:
        # The core cancels synchronously and its CANCELED update arrives through the listener.
        # Reporting True makes the base class apply CANCELED (idempotent) rather than
        # PENDING_CANCEL, which would regress the already-CANCELED order when it is applied last.
        return True

    @property
    def is_trading_required(self) -> bool:
        return True

    def supported_order_types(self) -> list[OrderType]:
        return self._core.supported_order_types()

    # --- Errors: nothing is requested over a network ---

    def _is_request_exception_related_to_time_synchronizer(self, request_exception: Exception) -> bool:
        return False

    def _is_order_not_found_during_status_update_error(self, status_update_exception: Exception) -> bool:
        return False

    def _is_order_not_found_during_cancelation_error(self, cancelation_exception: Exception) -> bool:
        return False

    # --- Factories ---

    def _create_web_assistants_factory(self) -> WebAssistantsFactory:
        return WebAssistantsFactory(throttler=self._throttler)

    def _create_order_book_data_source(self) -> OrderBookTrackerDataSource:
        return _NoOpOrderBookDataSource(trading_pairs=self.trading_pairs)

    def _create_user_stream_data_source(self) -> UserStreamTrackerDataSource:
        return _NoOpUserStreamDataSource()

    # --- Trading rules and symbols ---

    async def _format_trading_rules(self, exchange_info_dict: dict[str, Any]) -> list[TradingRule]:
        return [
            TradingRule(
                trading_pair=rule.trading_pair,
                min_order_size=rule.min_order_size,
                max_order_size=rule.max_order_size,
                min_price_increment=rule.min_price_increment,
                min_base_amount_increment=rule.min_base_amount_increment,
                min_quote_amount_increment=rule.min_quote_amount_increment,
                min_notional_size=rule.min_notional_size,
            )
            for rule in self._core.trading_rules.values()
        ]

    async def _update_trading_rules(self):
        await self._core.update_trading_rules()
        rules = await self._format_trading_rules({})
        self._trading_rules.clear()
        for rule in rules:
            self._trading_rules[rule.trading_pair] = rule
        self._set_trading_pair_symbol_map({rule.trading_pair: rule.trading_pair for rule in rules})

    def _initialize_trading_pair_symbols_from_exchange_info(self, exchange_info: dict[str, Any]):
        self._set_trading_pair_symbol_map({pair: pair for pair in self.trading_pairs})

    # --- Orders ---

    async def _place_order(
        self,
        order_id: str,
        trading_pair: str,
        amount: Decimal,
        trade_type: TradeType,
        order_type: OrderType,
        price: Decimal,
        **kwargs,
    ) -> tuple[str, float]:
        exchange_order_id = await self._core.place_order(
            client_order_id=order_id,
            trading_pair=trading_pair,
            amount=amount,
            trade_type=trade_type,
            order_type=order_type,
            price=price,
            **kwargs,
        )
        return exchange_order_id, self._core.sim_exchange.current_timestamp

    async def _place_cancel(self, order_id: str, tracked_order: InFlightOrder):
        return await self._core.cancel_order(order_id, tracked_order.trading_pair)

    def _get_fee(
        self,
        base_currency: str,
        quote_currency: str,
        order_type: OrderType,
        order_side: TradeType,
        amount: Decimal,
        price: Decimal = s_decimal_NaN,
        is_maker: bool | None = None,
    ) -> TradeFeeBase:
        fee = self._core.get_fee(base_currency, quote_currency, order_type, order_side, amount, price, is_maker)
        return _to_hb_fee(fee)

    async def _all_trade_updates_for_order(self, order: InFlightOrder) -> list[TradeUpdate]:
        updates = await self._core.all_trade_updates_for_order(order.client_order_id)
        return [_to_hb_trade_update(update) for update in updates]

    async def _request_order_status(self, tracked_order: InFlightOrder) -> OrderUpdate:
        update = await self._core.request_order_status(tracked_order.client_order_id, tracked_order.trading_pair)
        return _to_hb_order_update(update)

    # --- Balances / fees ---

    async def _update_balances(self):
        await self._core.update_balances()
        self._account_balances = self._core.get_all_balances()
        self._account_available_balances = self._core.get_all_available_balances()

    async def _update_trading_fees(self):
        """Fees are configured in the core, not fetched."""

    async def _user_stream_event_listener(self):
        """No user stream: fills and cancels arrive through the core listeners."""
        await _park_forever()

    # --- Clock ---

    def tick(self, timestamp: float):
        self._core.tick(timestamp)
        super().tick(timestamp)
