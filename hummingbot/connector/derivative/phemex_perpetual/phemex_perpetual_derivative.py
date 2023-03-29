import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any, AsyncIterable, Dict, List, Optional, Tuple

from hummingbot.connector.constants import s_decimal_NaN
from hummingbot.connector.derivative.phemex_perpetual import (
    phemex_perpetual_constants as CONSTANTS,
    phemex_perpetual_utils,
    phemex_perpetual_web_utils as web_utils,
)
from hummingbot.connector.derivative.phemex_perpetual.phemex_perpetual_api_order_book_data_source import (
    PhemexPerpetualAPIOrderBookDataSource,
)
from hummingbot.connector.derivative.phemex_perpetual.phemex_perpetual_api_user_stream_data_source import (
    PhemexPerpetualAPIUserStreamDataSource,
)
from hummingbot.connector.derivative.phemex_perpetual.phemex_perpetual_auth import PhemexPerpetualAuth
from hummingbot.connector.exchange_base import bidict
from hummingbot.connector.perpetual_derivative_py_base import PerpetualDerivativePyBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.connector.utils import combine_to_hb_trading_pair
from hummingbot.core.api_throttler.data_types import RateLimit
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState, OrderUpdate, TradeUpdate
from hummingbot.core.data_type.order_book_tracker_data_source import OrderBookTrackerDataSource
from hummingbot.core.data_type.trade_fee import TokenAmount, TradeFeeBase
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.utils.async_utils import safe_ensure_future, safe_gather
from hummingbot.core.utils.estimate_fee import build_trade_fee
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory

if TYPE_CHECKING:
    from hummingbot.client.config.config_helpers import ClientConfigAdapter

bpm_logger = None


class PhemexPerpetualDerivative(PerpetualDerivativePyBase):
    web_utils = web_utils
    SHORT_POLL_INTERVAL = 5.0
    UPDATE_ORDER_STATUS_MIN_INTERVAL = 10.0
    LONG_POLL_INTERVAL = 120.0

    def __init__(
        self,
        client_config_map: "ClientConfigAdapter",
        phemex_perpetual_api_key: str = None,
        phemex_perpetual_api_secret: str = None,
        trading_pairs: Optional[List[str]] = None,
        trading_required: bool = True,
        domain: str = CONSTANTS.DEFAULT_DOMAIN,
    ):
        self.phemex_perpetual_api_key = phemex_perpetual_api_key
        self.phemex_perpetual_secret_key = phemex_perpetual_api_secret
        self._trading_required = trading_required
        self._trading_pairs = trading_pairs
        self._domain = domain
        self._position_mode = None
        self._last_trade_history_timestamp = None
        super().__init__(client_config_map)

    @property
    def name(self) -> str:
        return CONSTANTS.EXCHANGE_NAME

    @property
    def authenticator(self) -> PhemexPerpetualAuth:
        return PhemexPerpetualAuth(
            self.phemex_perpetual_api_key, self.phemex_perpetual_secret_key, self._time_synchronizer
        )

    @property
    def rate_limits_rules(self) -> List[RateLimit]:
        return CONSTANTS.RATE_LIMITS

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def client_order_id_max_length(self) -> int:
        return CONSTANTS.MAX_ORDER_ID_LEN

    @property
    def client_order_id_prefix(self) -> str:
        return ""

    @property
    def trading_rules_request_path(self) -> str:
        return CONSTANTS.EXCHANGE_INFO_URL

    @property
    def trading_pairs_request_path(self) -> str:
        return CONSTANTS.EXCHANGE_INFO_URL

    @property
    def check_network_request_path(self) -> str:
        return CONSTANTS.SERVER_TIME_PATH_URL

    @property
    def trading_pairs(self):
        return self._trading_pairs

    @property
    def is_cancel_request_in_exchange_synchronous(self) -> bool:
        return False

    @property
    def is_trading_required(self) -> bool:
        return self._trading_required

    @property
    def funding_fee_poll_interval(self) -> int:
        return 120

    def supported_order_types(self) -> List[OrderType]:
        """
        :return a list of OrderType supported by this connector
        """
        return [OrderType.LIMIT, OrderType.LIMIT_MAKER]

    def supported_position_modes(self):
        """
        This method needs to be overridden to provide the accurate information depending on the exchange.
        """
        return [PositionMode.ONEWAY, PositionMode.HEDGE]

    def get_buy_collateral_token(self, trading_pair: str) -> str:
        trading_rule: TradingRule = self._trading_rules[trading_pair]
        return trading_rule.buy_order_collateral_token

    def get_sell_collateral_token(self, trading_pair: str) -> str:
        trading_rule: TradingRule = self._trading_rules[trading_pair]
        return trading_rule.sell_order_collateral_token

    def _is_request_exception_related_to_time_synchronizer(self, request_exception: Exception):
        """
        Documentation doesn't make this clear.
        To-do: Confirm manually or from their team.
        """
        return False

    def _is_order_not_found_during_status_update_error(self, status_update_exception: Exception) -> bool:
        return "Order not found for Client ID" in str(status_update_exception)

    def _is_order_not_found_during_cancelation_error(self, cancelation_exception: Exception) -> bool:
        return str(CONSTANTS.ORDER_NOT_FOUND_ERROR_CODE) in str(
            cancelation_exception
        ) and CONSTANTS.ORDER_NOT_FOUND_ERROR_MESSAGE in str(cancelation_exception)

    def _create_web_assistants_factory(self) -> WebAssistantsFactory:
        return web_utils.build_api_factory(
            throttler=self._throttler, time_synchronizer=self._time_synchronizer, domain=self._domain, auth=self._auth
        )

    def _create_order_book_data_source(self) -> OrderBookTrackerDataSource:
        return PhemexPerpetualAPIOrderBookDataSource(
            trading_pairs=self._trading_pairs,
            connector=self,
            api_factory=self._web_assistants_factory,
            domain=self.domain,
        )

    def _create_user_stream_data_source(self) -> UserStreamTrackerDataSource:
        return PhemexPerpetualAPIUserStreamDataSource(
            auth=self._auth,
            api_factory=self._web_assistants_factory,
            domain=self.domain,
        )

    def _get_fee(
        self,
        base_currency: str,
        quote_currency: str,
        order_type: OrderType,
        order_side: TradeType,
        amount: Decimal,
        price: Decimal = s_decimal_NaN,
        is_maker: Optional[bool] = None,
    ) -> TradeFeeBase:
        is_maker = is_maker or False
        fee = build_trade_fee(
            self.name,
            is_maker,
            base_currency=base_currency,
            quote_currency=quote_currency,
            order_type=order_type,
            order_side=order_side,
            amount=amount,
            price=price,
        )
        return fee

    async def _update_trading_fees(self):
        """
        Update fees information from the exchange
        """
        raise NotImplementedError()

    async def _status_polling_loop_fetch_updates(self):
        await safe_gather(
            self._update_order_status(),
            """To-Do: Update
            self._update_balances(),
            self._update_positions(),""",
        )

    async def _place_cancel(self, order_id: str, tracked_order: InFlightOrder):
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair=tracked_order.trading_pair)
        if self._position_mode is PositionMode.ONEWAY:
            posSide = "Merged"
        else:
            posSide = "Long" if tracked_order.trade_type is TradeType.BUY else "Short"
        api_params = {"clOrdID": order_id, "symbol": symbol, "posSide": posSide}
        cancel_result = await self._api_delete(
            path_url=CONSTANTS.CANCEL_ORDERS, params=api_params, is_auth_required=True
        )

        if cancel_result["code"] != CONSTANTS.SUCCESSFUL_RETURN_CODE:
            code = cancel_result["code"]
            message = cancel_result["msg"]
            raise IOError(f"{code} - {message}")
        is_order_canceled = CONSTANTS.ORDER_STATE[cancel_result["data"]["ordStatus"]] == OrderState.CANCELED

        return is_order_canceled

    async def _place_order(
        self,
        order_id: str,
        trading_pair: str,
        amount: Decimal,
        trade_type: TradeType,
        order_type: OrderType,
        price: Decimal,
        position_action: PositionAction = PositionAction.NIL,
        **kwargs,
    ) -> Tuple[str, float]:

        amount_str = f"{amount:f}"
        price_str = f"{price:f}"
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
        api_params = {
            "symbol": symbol,
            "side": "Buy" if trade_type is TradeType.BUY else "Sell",
            "orderQtyRq": amount_str,
            "ordType": "Market" if order_type is OrderType.MARKET else "Limit",
            "clOrdID": order_id,
            "closeOnTrigger": position_action == PositionAction.CLOSE,
            "reduceOnly": position_action == PositionAction.CLOSE,
        }
        if order_type.is_limit_type():
            api_params["priceRp"] = price_str
        if order_type == OrderType.LIMIT_MAKER:
            api_params["timeInForce"] = "PostOnly"
        if self._position_mode is PositionMode.ONEWAY:
            api_params["posSide"] = "Merged"
        else:
            if position_action == PositionAction.OPEN:
                api_params["posSide"] = "Long" if trade_type is TradeType.BUY else "Short"
            else:
                api_params["posSide"] = "Short" if trade_type is TradeType.BUY else "Long"

        order_result = await self._api_post(path_url=CONSTANTS.PLACE_ORDERS, data=api_params, is_auth_required=True)
        o_id = str(order_result["data"]["orderID"])
        transact_time = order_result["data"]["actionTimeNs"] * 1e-9
        return o_id, transact_time

    async def _all_trade_updates_for_order(self, order: InFlightOrder) -> List[TradeUpdate]:
        # Not required in Phemex because it reimplements _update_orders_fills
        raise NotImplementedError

    async def _all_trades_details(self, trading_pair: str, start_time: float) -> List[Dict[str, Any]]:
        result = []
        try:
            symbol = await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
            result = await self._api_get(
                path_url=CONSTANTS.GET_TRADES,
                params={"symbol": symbol, "start": int(start_time * 1e3), "limit": 200},
                is_auth_required=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            self.logger().warning(f"There was an error requesting trades history for Phemex ({ex})")

        return result

    async def _update_orders_fills(self, orders: List[InFlightOrder]):
        # Reimplementing this method because Phemex does not provide an endpoint to request trades for a particular
        # order

        if len(orders) > 0:
            orders_by_id = dict()
            min_order_creation_time = self.current_timestamp
            trading_pairs = set()

            for order in orders:
                orders_by_id[order.client_order_id] = order
                trading_pairs.add(order.trading_pair)
                min_order_creation_time = min(min_order_creation_time, order.creation_timestamp)

            tasks = [
                safe_ensure_future(
                    self._all_trades_details(trading_pair=trading_pair, start_time=min_order_creation_time)
                ) for trading_pair in trading_pairs
            ]

            trades_data = []
            results = await safe_gather(*tasks)
            for trades_for_market in results:
                trades_data.extend(trades_for_market)

            for trade_info in trades_data:
                client_order_id = trade_info["clOrdID"]
                tracked_order = orders_by_id.get(client_order_id)

                if tracked_order is not None:
                    position_action = tracked_order.position
                    fee = TradeFeeBase.new_perpetual_fee(
                        fee_schema=self.trade_fee_schema(),
                        position_action=position_action,
                        percent_token=CONSTANTS.COLLATERAL_TOKEN,
                        flat_fees=[TokenAmount(
                            amount=Decimal(trade_info["execFeeRv"]),
                            token=CONSTANTS.COLLATERAL_TOKEN)
                        ],
                    )
                    trade_update: TradeUpdate = TradeUpdate(
                        trade_id=trade_info["execID"],
                        client_order_id=tracked_order.client_order_id,
                        exchange_order_id=trade_info["orderID"],
                        trading_pair=tracked_order.trading_pair,
                        fill_timestamp=trade_info["transactTimeNs"] * 1e-9,
                        fill_price=Decimal(trade_info["execPriceRp"]),
                        fill_base_amount=Decimal(trade_info["execQtyRq"]),
                        fill_quote_amount=Decimal(trade_info["execValueRv"]),
                        fee=fee,
                    )

                    self._order_tracker.process_trade_update(trade_update=trade_update)

    async def _request_order_status(self, tracked_order: InFlightOrder) -> OrderUpdate:
        trading_pair = await self.exchange_symbol_associated_to_pair(trading_pair=tracked_order.trading_pair)
        response = await self._api_get(
            path_url=CONSTANTS.GET_ORDERS,
            params={"symbol": trading_pair, "clOrdID": tracked_order.client_order_id},
            is_auth_required=True,
        )

        orders_data = response.get("data", {}).get("rows", [])

        if len(orders_data) == 0:
            raise IOError(f"Order not found for Client ID {tracked_order.client_order_id}")

        order_info = orders_data[0]
        order_update: OrderUpdate = OrderUpdate(
            trading_pair=tracked_order.trading_pair,
            update_timestamp=self.current_timestamp,
            new_state=CONSTANTS.ORDER_STATE[order_info["ordStatus"]],
            client_order_id=tracked_order.client_order_id,
            exchange_order_id=order_info["orderId"],
        )
        return order_update

    async def _iter_user_event_queue(self) -> AsyncIterable[Dict[str, any]]:
        while True:
            try:
                yield await self._user_stream_tracker.user_stream.get()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger().network(
                    "Unknown error. Retrying after 1 seconds.",
                    exc_info=True,
                    app_warning_msg="Could not fetch user events from Phemex. Check API key and network connection.",
                )
                await self._sleep(1.0)

    async def _user_stream_event_listener(self):
        """
        Wait for new messages from _user_stream_tracker.user_stream queue and processes them according to their
        message channels. The respective UserStreamDataSource queues these messages.
        """
        async for event_message in self._iter_user_event_queue():
            try:
                await self._process_user_stream_event(event_message)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger().error(f"Unexpected error in user stream listener loop: {e}", exc_info=True)
                await self._sleep(5.0)

    async def _process_user_stream_event(self, event_message: Dict[str, Any]):
        event_type = event_message.get("e")
        if event_type == "ORDER_TRADE_UPDATE":

            """trade_update: TradeUpdate = TradeUpdate(

                        )
                        self._order_tracker.process_trade_update(trade_update)
            AND
                        tracked_order = self._order_tracker.all_updatable_orders.get(client_order_id)
                        if tracked_order is not None:
                            order_update: OrderUpdate = OrderUpdate(

                            )

                            self._order_tracker.process_order_update(order_update)"""

        elif event_type == "ACCOUNT_UPDATE":
            """

                        position.update_position(position_side=PositionSide[asset["ps"]],
                                                    unrealized_pnl=Decimal(asset["up"]),
                                                    entry_price=Decimal(asset["ep"]),
                                                    amount=Decimal(asset["pa"]))
            OR
                        await self._update_positions()
            """

        elif event_type == "MARGIN_CALL":
            """self.logger().warning("Margin Call: Your position risk is too high, and you are at risk of "
                                  "liquidation. Close your positions or add additional margin to your wallet.")
            self.logger().info(f"Margin Required: {total_maint_margin_required}. "
                               f"Negative PnL assets: {negative_pnls_msg}.")"""

    async def _format_trading_rules(self, exchange_info_dict: Dict[str, Any]) -> List[TradingRule]:
        """
        Queries the necessary API endpoint and initialize the TradingRule object for each trading pair being traded.

        Parameters
        ----------
        exchange_info_dict:
            Trading rules dictionary response from the exchange
        """
        raise NotImplementedError()

    def _initialize_trading_pair_symbols_from_exchange_info(self, exchange_info: Dict[str, Any]):
        mapping = bidict()
        for symbol_data in filter(
                phemex_perpetual_utils.is_exchange_information_valid,
                exchange_info["data"]["perpProductsV2"],
        ):
            exchange_symbol = symbol_data["symbol"]
            base = symbol_data["contractUnderlyingAssets"]
            quote = symbol_data["settleCurrency"]
            trading_pair = combine_to_hb_trading_pair(base, quote)
            mapping[exchange_symbol] = trading_pair
        self._set_trading_pair_symbol_map(mapping)

    async def _update_balances(self):
        """
        Calls the REST API to update total and available balances.
        """
        account_info = await self._api_get(
            path_url=CONSTANTS.ACCOUNT_INFO,
            params={"currency": CONSTANTS.COLLATERAL_TOKEN},
            is_auth_required=True,
        )

        if account_info["code"] != CONSTANTS.SUCCESSFUL_RETURN_CODE:
            code = account_info["code"]
            message = account_info["msg"]
            raise IOError(f"{code} - {message}")

        account_data = account_info["data"]["account"]

        self._account_available_balances.clear()
        self._account_balances.clear()
        total_balance = Decimal(str(account_data["accountBalanceRv"]))
        locked_balance = Decimal(str(account_data["totalUsedBalanceRv"]))
        self._account_balances[account_data["currency"]] = total_balance
        self._account_available_balances[account_data["currency"]] = total_balance - locked_balance

    async def _update_positions(self):
        """
        if _position:
            self._perpetual_trading.set_position(pos_key, _position)
        else:
            self._perpetual_trading.remove_position(pos_key)
        """
        raise NotImplementedError()

    async def _get_position_mode(self) -> Optional[PositionMode]:
        # To-do:
        pass

    async def _trading_pair_position_mode_set(self, mode: PositionMode, trading_pair: str) -> Tuple[bool, str]:
        raise NotImplementedError()

    async def _set_trading_pair_leverage(self, trading_pair: str, leverage: int) -> Tuple[bool, str]:
        raise NotImplementedError()

    async def _fetch_last_fee_payment(self, trading_pair: str) -> Tuple[int, Decimal, Decimal]:
        raise NotImplementedError()
