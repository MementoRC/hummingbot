import logging
from decimal import Decimal
from typing import List

import pandas as pd

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.derivative_base import DerivativeBase
from hummingbot.core.data_type.common import OrderType, PositionAction
from hummingbot.core.data_type.trade import Trade
from hummingbot.core.event.event_listener import EventListener
from hummingbot.core.event.events import AccountEvent, MarketEvent, OrderFilledEvent
from hummingbot.core.network_iterator import NetworkStatus
from hummingbot.core.time_iterator import TimeIterator
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.order_tracker import OrderTracker

NaN = float("nan")
s_decimal_nan = Decimal("NaN")
s_decimal_0 = Decimal("0")

# <editor-fold desc="+ Event listeners">


class BaseStrategyEventListener(EventListener):
    def __init__(self, owner: "StrategyBase"):
        super().__init__()
        self._owner = owner


class BuyOrderCompletedListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_complete_buy_order(arg)
        self._owner.c_did_complete_buy_order_tracker(arg)


class SellOrderCompletedListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_complete_sell_order(arg)
        self._owner.c_did_complete_sell_order_tracker(arg)


class FundingPaymentCompletedListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_complete_funding_payment(arg)


class PositionModeChangeSuccessListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_change_position_mode_succeed(arg)


class PositionModeChangeFailureListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_change_position_mode_fail(arg)


class OrderFilledListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_fill_order(arg)


class OrderFailedListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_fail_order(arg)
        self._owner.c_did_fail_order_tracker(arg)


class OrderCancelledListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_cancel_order(arg)
        self._owner.c_did_cancel_order_tracker(arg)


class OrderExpiredListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_expire_order(arg)
        self._owner.c_did_expire_order_tracker(arg)


class BuyOrderCreatedListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_create_buy_order(arg)


class SellOrderCreatedListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_create_sell_order(arg)


class RangePositionLiquidityAddedListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_add_liquidity(arg)


class RangePositionLiquidityRemovedListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_remove_liquidity(arg)


class RangePositionUpdateFailureListener(BaseStrategyEventListener):
    def __call__(self, arg: object) -> None:
        self._owner.c_did_fail_lp_update(arg)


# </editor-fold>


class StrategyBase(TimeIterator):
    BUY_ORDER_COMPLETED_EVENT_TAG = MarketEvent.BuyOrderCompleted.value
    SELL_ORDER_COMPLETED_EVENT_TAG = MarketEvent.SellOrderCompleted.value
    FUNDING_PAYMENT_COMPLETED_EVENT_TAG = MarketEvent.FundingPaymentCompleted.value
    POSITION_MODE_CHANGE_SUCCEEDED_EVENT_TAG = AccountEvent.PositionModeChangeSucceeded.value
    POSITION_MODE_CHANGE_FAILED_EVENT_TAG = AccountEvent.PositionModeChangeFailed.value
    ORDER_FILLED_EVENT_TAG = MarketEvent.OrderFilled.value
    ORDER_CANCELED_EVENT_TAG = MarketEvent.OrderCancelled.value
    ORDER_EXPIRED_EVENT_TAG = MarketEvent.OrderExpired.value
    ORDER_FAILURE_EVENT_TAG = MarketEvent.OrderFailure.value
    BUY_ORDER_CREATED_EVENT_TAG = MarketEvent.BuyOrderCreated.value
    SELL_ORDER_CREATED_EVENT_TAG = MarketEvent.SellOrderCreated.value
    RANGE_POSITION_LIQUIDITY_ADDED_EVENT_TAG = MarketEvent.RangePositionLiquidityAdded.value
    RANGE_POSITION_LIQUIDITY_REMOVED_EVENT_TAG = MarketEvent.RangePositionLiquidityRemoved.value
    RANGE_POSITION_UPDATE_FAILURE_EVENT_TAG = MarketEvent.RangePositionUpdateFailure.value

    @classmethod
    def logger(cls) -> logging.Logger:
        raise NotImplementedError

    def __init__(self):
        super().__init__()
        self._sb_markets: set = set()
        self._sb_create_buy_order_listener: EventListener = BuyOrderCreatedListener(self)
        self._sb_create_sell_order_listener: EventListener = SellOrderCreatedListener(self)
        self._sb_fill_order_listener: EventListener = OrderFilledListener(self)
        self._sb_fail_order_listener: EventListener = OrderFailedListener(self)
        self._sb_cancel_order_listener: EventListener = OrderCancelledListener(self)
        self._sb_expire_order_listener: EventListener = OrderExpiredListener(self)
        self._sb_complete_buy_order_listener: EventListener = BuyOrderCompletedListener(self)
        self._sb_complete_sell_order_listener: EventListener = SellOrderCompletedListener(self)
        self._sb_complete_funding_payment_listener: EventListener = FundingPaymentCompletedListener(self)
        self._sb_position_mode_change_success_listener: EventListener = PositionModeChangeSuccessListener(self)
        self._sb_position_mode_change_failure_listener: EventListener = PositionModeChangeFailureListener(self)
        self._sb_range_position_liquidity_added_listener: EventListener = RangePositionLiquidityAddedListener(self)
        self._sb_range_position_liquidity_removed_listener: EventListener = RangePositionLiquidityRemovedListener(self)
        self._sb_range_position_update_failure_listener: EventListener = RangePositionUpdateFailureListener(self)
        self._sb_delegate_lock: bool = False
        self._sb_order_tracker: OrderTracker = OrderTracker()

    def init_params(self, *args, **kwargs):
        """
        Assigns strategy parameters, this function must be called directly after init.
        The reason for this is to make the parameters discoverable through introspect (this is not possible on init of
        a Cython class).
        """
        raise NotImplementedError

    @property
    def active_markets(self) -> List[ConnectorBase]:
        return list(self._sb_markets)

    @property
    def order_tracker(self) -> OrderTracker:
        return self._sb_order_tracker

    def format_status(self):
        raise NotImplementedError

    def log_with_clock(self, log_level: int, msg: str, **kwargs):
        clock_timestamp = pd.Timestamp(self.current_timestamp, unit="s", tz="UTC")
        self.logger().log(log_level, f"{msg} [clock={str(clock_timestamp)}]", **kwargs)

    @property
    def trades(self) -> List[Trade]:
        """
        Returns a list of all completed trades from the market.
        The trades are taken from the market event logs.
        """

        def event_to_trade(order_filled_event: OrderFilledEvent, market_name: str):
            return Trade(
                order_filled_event.trading_pair,
                order_filled_event.trade_type,
                order_filled_event.price,
                order_filled_event.amount,
                order_filled_event.order_type,
                market_name,
                order_filled_event.timestamp,
                order_filled_event.trade_fee,
            )

        past_trades = []
        for market in self.active_markets:
            event_logs = market.event_logs
            order_filled_events = list(filter(lambda e: isinstance(e, OrderFilledEvent), event_logs))
            past_trades += list(map(lambda ofe: event_to_trade(ofe, market.display_name), order_filled_events))

        return sorted(past_trades, key=lambda x: x.timestamp)

    def market_status_data_frame(self, market_trading_pair_tuples: List[MarketTradingPairTuple]) -> pd.DataFrame:
        markets_data = []
        markets_columns = ["Exchange", "Market", "Best Bid Price", "Best Ask Price", "Mid Price"]
        try:
            for market_trading_pair_tuple in market_trading_pair_tuples:
                market, trading_pair, base_asset, quote_asset = market_trading_pair_tuple
                bid_price = market.get_price(trading_pair, False)
                ask_price = market.get_price(trading_pair, True)
                mid_price = (bid_price + ask_price) / 2
                markets_data.append(
                    [market.display_name, trading_pair, float(bid_price), float(ask_price), float(mid_price)]
                )
            return pd.DataFrame(data=markets_data, columns=markets_columns)

        except Exception:
            self.logger().error("Error formatting market stats.", exc_info=True)

    def wallet_balance_data_frame(self, market_trading_pair_tuples: List[MarketTradingPairTuple]) -> pd.DataFrame:
        assets_data = []
        assets_columns = ["Exchange", "Asset", "Total Balance", "Available Balance"]
        try:
            for market_trading_pair_tuple in market_trading_pair_tuples:
                market, trading_pair, base_asset, quote_asset = market_trading_pair_tuple
                base_balance = float(market.get_balance(base_asset))
                quote_balance = float(market.get_balance(quote_asset))
                available_base_balance = float(market.get_available_balance(base_asset))
                available_quote_balance = float(market.get_available_balance(quote_asset))
                assets_data.extend(
                    [
                        [market.display_name, base_asset, base_balance, available_base_balance],
                        [market.display_name, quote_asset, quote_balance, available_quote_balance],
                    ]
                )

            return pd.DataFrame(data=assets_data, columns=assets_columns)

        except Exception:
            self.logger().error("Error formatting wallet balance stats.", exc_info=True)

    def balance_warning(self, market_trading_pair_tuples: List[MarketTradingPairTuple]) -> List[str]:
        warning_lines = []
        # Add warning lines on null balances.
        # TO-DO: $Use min order size logic to replace the hard-coded 0.0001 value for each asset.
        for market_trading_pair_tuple in market_trading_pair_tuples:
            base_balance = market_trading_pair_tuple.market.get_balance(market_trading_pair_tuple.base_asset)
            quote_balance = market_trading_pair_tuple.market.get_balance(market_trading_pair_tuple.quote_asset)
            if base_balance <= Decimal("0.0001") and not isinstance(market_trading_pair_tuple.market, DerivativeBase):
                warning_lines.append(
                    f"  {market_trading_pair_tuple.market.name} market "
                    f"{market_trading_pair_tuple.base_asset} balance is too low. Cannot place order."
                )
            if quote_balance <= Decimal("0.0001"):
                warning_lines.append(
                    f"  {market_trading_pair_tuple.market.name} market "
                    f"{market_trading_pair_tuple.quote_asset} balance is too low. Cannot place order."
                )
        return warning_lines

    def network_warning(self, market_trading_pair_tuples: List[MarketTradingPairTuple]) -> List[str]:
        warning_lines = []
        if not all(
            [
                market_trading_pair_tuple.market.network_status is NetworkStatus.CONNECTED
                for market_trading_pair_tuple in market_trading_pair_tuples
            ]
        ):
            trading_pairs = " // ".join(
                [market_trading_pair_tuple.trading_pair for market_trading_pair_tuple in market_trading_pair_tuples]
            )
            warning_lines.extend(
                [
                    f"  Markets are offline for the {trading_pairs} pair. Continued trading "
                    f"with these markets may be dangerous.",
                    "",
                ]
            )
        return warning_lines

    def c_start(self, clock, timestamp: float) -> None:
        TimeIterator.c_start(self, clock, timestamp)
        OrderTracker.c_start(self._sb_order_tracker, clock, timestamp)

    def c_tick(self, timestamp: float) -> None:
        TimeIterator.c_tick(self, timestamp)
        OrderTracker.c_tick(self._sb_order_tracker, timestamp)

    def c_stop(self, clock) -> None:
        TimeIterator.c_stop(self, clock)
        OrderTracker.c_stop(self._sb_order_tracker, clock)
        self.c_remove_markets(list(self._sb_markets))

    def c_add_markets(self, markets: list) -> None:
        for market in markets:
            market.add_listener(MarketEvent.BuyOrderCreated, self._sb_create_buy_order_listener)
            market.add_listener(MarketEvent.SellOrderCreated, self._sb_create_sell_order_listener)
            market.add_listener(MarketEvent.OrderFilled, self._sb_fill_order_listener)
            market.add_listener(MarketEvent.OrderFailure, self._sb_fail_order_listener)
            market.add_listener(MarketEvent.OrderCancelled, self._sb_cancel_order_listener)
            market.add_listener(MarketEvent.OrderExpired, self._sb_expire_order_listener)
            market.add_listener(MarketEvent.BuyOrderCompleted, self._sb_complete_buy_order_listener)
            market.add_listener(MarketEvent.SellOrderCompleted, self._sb_complete_sell_order_listener)
            market.add_listener(MarketEvent.FundingPaymentCompleted, self._sb_complete_funding_payment_listener)
            market.add_listener(
                AccountEvent.PositionModeChangeSucceeded, self._sb_position_mode_change_success_listener
            )
            market.add_listener(AccountEvent.PositionModeChangeFailed, self._sb_position_mode_change_failure_listener)
            market.add_listener(
                MarketEvent.RangePositionLiquidityAdded, self._sb_range_position_liquidity_added_listener
            )
            market.add_listener(
                MarketEvent.RangePositionLiquidityRemoved, self._sb_range_position_liquidity_removed_listener
            )
            market.add_listener(MarketEvent.RangePositionUpdateFailure, self._sb_range_position_update_failure_listener)
            self._sb_markets.add(market)

    def add_markets(self, markets: List[ConnectorBase]) -> None:
        self.c_add_markets(markets)

    def c_remove_markets(self, markets: list) -> None:
        for market in markets:
            if market not in self._sb_markets:
                continue
            market.remove_listener(MarketEvent.BuyOrderCreated, self._sb_create_buy_order_listener)
            market.remove_listener(MarketEvent.SellOrderCreated, self._sb_create_sell_order_listener)
            market.remove_listener(MarketEvent.OrderFilled, self._sb_fill_order_listener)
            market.remove_listener(MarketEvent.OrderFailure, self._sb_fail_order_listener)
            market.remove_listener(MarketEvent.OrderCancelled, self._sb_cancel_order_listener)
            market.remove_listener(MarketEvent.OrderExpired, self._sb_expire_order_listener)
            market.remove_listener(MarketEvent.BuyOrderCompleted, self._sb_complete_buy_order_listener)
            market.remove_listener(MarketEvent.SellOrderCompleted, self._sb_complete_sell_order_listener)
            market.remove_listener(MarketEvent.FundingPaymentCompleted, self._sb_complete_funding_payment_listener)
            market.remove_listener(
                AccountEvent.PositionModeChangeSucceeded, self._sb_position_mode_change_success_listener
            )
            market.remove_listener(
                AccountEvent.PositionModeChangeFailed, self._sb_position_mode_change_failure_listener
            )
            market.remove_listener(
                MarketEvent.RangePositionLiquidityAdded, self._sb_range_position_liquidity_added_listener
            )
            market.remove_listener(
                MarketEvent.RangePositionLiquidityRemoved, self._sb_range_position_liquidity_removed_listener
            )
            market.remove_listener(
                MarketEvent.RangePositionUpdateFailure, self._sb_range_position_update_failure_listener
            )
            self._sb_markets.remove(market)

    def remove_markets(self, markets: List[ConnectorBase]) -> None:
        self.c_remove_markets(markets)

    def c_sum_flat_fees(self, quote_asset: str, flat_fees: list) -> Decimal:
        """
        Converts flat fees to quote token and sums up all flat fees
        """
        total_flat_fees = s_decimal_0
        for flat_fee_currency, flat_fee_amount in flat_fees:
            if flat_fee_currency == quote_asset:
                total_flat_fees += flat_fee_amount
            else:
                # if the flat fee currency asset does not match quote asset, raise exception for now
                # as we don't support different token conversion atm.
                raise Exception("Flat fee in other token than quote asset is not supported.")
        return total_flat_fees

    def cum_flat_fees(self, quote_asset: str, flat_fees: List) -> Decimal:
        return self.c_sum_flat_fees(quote_asset, flat_fees)

    # <editor-fold desc="+ Market event interfaces">
    # ----------------------------------------------------------------------------------------------------------
    def c_did_create_buy_order(self, order_created_event: object) -> None:
        """
        In the case of asynchronous order creation on the exchange's server, this event is NOT triggered
        upon submission of the order request to the server - it is only triggered once the server has sent
        the acknowledgment that the order is successfully created.
        """
        pass

    def c_did_create_sell_order(self, order_created_event: object) -> None:
        """
        In the case of asynchronous order creation on the exchange's server, this event is NOT triggered
        upon submission of the order request to the server - it is only triggered once the server has sent
        the acknowledgment that the order is successfully created.
        """
        pass

    def c_did_fill_order(self, order_filled_event: object) -> None:
        pass

    def c_did_fail_order(self, order_failed_event: object) -> None:
        pass

    def c_did_cancel_order(self, cancelled_event: object) -> None:
        pass

    def c_did_expire_order(self, expired_event: object) -> None:
        pass

    def c_did_complete_buy_order(self, order_completed_event: object) -> None:
        pass

    def c_did_complete_sell_order(self, order_completed_event: object) -> None:
        pass

    def c_did_complete_funding_payment(self, funding_payment_completed_event: object) -> None:
        pass

    def c_did_change_position_mode_succeed(self, position_mode_changed_event: object) -> None:
        pass

    def c_did_change_position_mode_fail(self, position_mode_changed_event: object) -> None:
        pass

    def c_did_add_liquidity(self, add_liquidity_event: object) -> None:
        pass

    def c_did_remove_liquidity(self, remove_liquidity_event: object) -> None:
        pass

    def c_did_fail_lp_update(self, fail_lp_update_event: object) -> None:
        pass

    # ----------------------------------------------------------------------------------------------------------
    # </editor-fold>

    # <editor-fold desc="+ Order tracking event handlers">
    # ----------------------------------------------------------------------------------------------------------
    def c_did_fail_order_tracker(self, order_failed_event: object) -> None:
        order_id: str = order_failed_event.order_id
        order_type = order_failed_event.order_type
        market_pair = self._sb_order_tracker.get_market_pair_from_order_id(order_id)

        if order_type.is_limit_type():
            self.c_stop_tracking_limit_order(market_pair, order_id)
        elif order_type == OrderType.MARKET:
            self.c_stop_tracking_market_order(market_pair, order_id)

    def c_did_cancel_order_tracker(self, order_cancelled_event: object) -> None:
        order_id: str = order_cancelled_event.order_id
        market_pair = self._sb_order_tracker.get_market_pair_from_order_id(order_id)
        self.c_stop_tracking_limit_order(market_pair, order_id)

    def c_did_expire_order_tracker(self, order_expired_event: object) -> None:
        self.c_did_cancel_order_tracker(order_expired_event)

    def c_did_complete_buy_order_tracker(self, order_completed_event: object) -> None:
        order_id: str = order_completed_event.order_id
        market_pair = self._sb_order_tracker.get_market_pair_from_order_id(order_id)
        order_type = order_completed_event.order_type

        if market_pair is not None:
            if order_type.is_limit_type():
                self.c_stop_tracking_limit_order(market_pair, order_id)
            elif order_type == OrderType.MARKET:
                self.c_stop_tracking_market_order(market_pair, order_id)

    def c_did_complete_sell_order_tracker(self, order_completed_event: object) -> None:
        self.c_did_complete_buy_order_tracker(order_completed_event)

    # ----------------------------------------------------------------------------------------------------------
    # </editor-fold>

    # <editor-fold desc="+ Creating and canceling orders">
    # ----------------------------------------------------------------------------------------------------------

    def buy_with_specific_market(
        self,
        market_trading_pair_tuple,
        amount,
        order_type=OrderType.MARKET,
        price=s_decimal_nan,
        expiration_seconds=NaN,
        position_action=PositionAction.OPEN,
    ):
        return self.c_buy_with_specific_market(
            market_trading_pair_tuple, amount, order_type, price, expiration_seconds, position_action
        )

    def c_buy_with_specific_market(
        self,
        market_trading_pair_tuple,
        amount,
        order_type=OrderType.MARKET,
        price=s_decimal_nan,
        expiration_seconds=NaN,
        position_action=PositionAction.OPEN,
    ) -> str:
        if self._sb_delegate_lock:
            raise RuntimeError("Delegates are not allowed to execute orders directly.")

        if not (isinstance(amount, Decimal) and isinstance(price, Decimal)):
            raise TypeError("price and amount must be Decimal objects.")

        kwargs = {
            "expiration_ts": self.current_timestamp + float(expiration_seconds),
            "position_action": position_action,
        }
        market: ConnectorBase = market_trading_pair_tuple.market

        if market not in self._sb_markets:
            raise ValueError("Market object for buy order is not in the whitelisted markets set.")

        order_id: str = market.buy(
            market_trading_pair_tuple.trading_pair,
            amount=amount,
            order_type=order_type,
            price=price,
            **kwargs,
        )

        # Start order tracking
        if order_type.is_limit_type():
            self.c_start_tracking_limit_order(market_trading_pair_tuple, order_id, True, price, amount)
        elif order_type == OrderType.MARKET:
            self.c_start_tracking_market_order(market_trading_pair_tuple, order_id, True, amount)

        return order_id

    def sell_with_specific_market(
        self,
        market_trading_pair_tuple,
        amount,
        order_type=OrderType.MARKET,
        price=s_decimal_nan,
        expiration_seconds=NaN,
        position_action=PositionAction.OPEN,
    ):
        return self.c_sell_with_specific_market(
            market_trading_pair_tuple, amount, order_type, price, expiration_seconds, position_action
        )

    def c_sell_with_specific_market(
        self,
        market_trading_pair_tuple,
        amount,
        order_type=OrderType.MARKET,
        price=s_decimal_nan,
        expiration_seconds=NaN,
        position_action=PositionAction.OPEN,
    ) -> str:
        if self._sb_delegate_lock:
            raise RuntimeError("Delegates are not allowed to execute orders directly.")

        if not (isinstance(amount, Decimal) and isinstance(price, Decimal)):
            raise TypeError("price and amount must be Decimal objects.")

        kwargs = {
            "expiration_ts": self.current_timestamp + float(expiration_seconds),
            "position_action": position_action,
        }
        market: ConnectorBase = market_trading_pair_tuple.market

        if market not in self._sb_markets:
            raise ValueError("Market object for sell order is not in the whitelisted markets set.")

        order_id: str = market.sell(
            market_trading_pair_tuple.trading_pair,
            amount,
            order_type=order_type,
            price=price,
            **kwargs,
        )

        # Start order tracking
        if order_type.is_limit_type():
            self.c_start_tracking_limit_order(market_trading_pair_tuple, order_id, False, price, amount)
        elif order_type == OrderType.MARKET:
            self.c_start_tracking_market_order(market_trading_pair_tuple, order_id, False, amount)

        return order_id

    def c_cancel_order(self, market_trading_pair_tuple, order_id: str) -> None:
        market: ConnectorBase = market_trading_pair_tuple.market

        if self._sb_order_tracker.check_and_track_cancel(order_id):
            self.log_with_clock(
                logging.INFO, f"({market_trading_pair_tuple.trading_pair}) Canceling the limit order {order_id}."
            )
            market.cancel(market_trading_pair_tuple.trading_pair, order_id)

    def cancel_order(self, market_trading_pair_tuple: MarketTradingPairTuple, order_id: str) -> None:
        self.c_cancel_order(market_trading_pair_tuple, order_id)

    # ----------------------------------------------------------------------------------------------------------
    # </editor-fold>

    # <editor-fold desc="+ Order tracking entry points">
    # The following exposed tracking functions are meant to allow extending order tracking behavior in strategy
    # classes.
    # ----------------------------------------------------------------------------------------------------------
    def c_start_tracking_limit_order(self, market_pair, order_id: str, is_buy: bool, price, quantity) -> None:
        self._sb_order_tracker.start_tracking_limit_order(market_pair, order_id, is_buy, price, quantity)

    def start_tracking_limit_order(
        self,
        market_pair: MarketTradingPairTuple,
        order_id: str,
        is_buy: bool,
        price: Decimal,
        quantity: Decimal,
    ) -> None:
        self.c_start_tracking_limit_order(market_pair, order_id, is_buy, price, quantity)

    def c_stop_tracking_limit_order(self, market_pair, order_id: str) -> None:
        self._sb_order_tracker.stop_tracking_limit_order(market_pair, order_id)

    def stop_tracking_limit_order(self, market_pair: MarketTradingPairTuple, order_id: str) -> None:
        self.c_stop_tracking_limit_order(market_pair, order_id)

    def c_start_tracking_market_order(self, market_pair, order_id: str, is_buy: bool, quantity) -> None:
        self._sb_order_tracker.start_tracking_market_order(market_pair, order_id, is_buy, quantity)

    def start_tracking_market_order(
        self, market_pair: MarketTradingPairTuple, order_id: str, is_buy: bool, quantity: Decimal
    ) -> None:
        self.c_start_tracking_market_order(market_pair, order_id, is_buy, quantity)

    def c_stop_tracking_market_order(self, market_pair, order_id: str) -> None:
        self._sb_order_tracker.stop_tracking_market_order(market_pair, order_id)

    def stop_tracking_market_order(self, market_pair: MarketTradingPairTuple, order_id: str) -> None:
        self.c_stop_tracking_market_order(market_pair, order_id)

    def c_track_restored_orders(self, market_pair) -> list:
        limit_orders = market_pair.market.limit_orders
        restored_order_ids = []

        for order in limit_orders:
            restored_order_ids.append(order.client_order_id)
            self.c_start_tracking_limit_order(
                market_pair,
                order.client_order_id,
                order.is_buy,
                order.price,
                order.quantity,
            )
        return restored_order_ids

    def track_restored_orders(self, market_pair: MarketTradingPairTuple) -> list:
        return self.c_track_restored_orders(market_pair)

    def notify_hb_app(self, msg: str) -> None:
        """
        Method called to display message on the Output Panel(upper left)
        :param msg: The message to be notified
        """
        from hummingbot.client.hummingbot_application import HummingbotApplication

        HummingbotApplication.main_application().notify(msg)

    def notify_hb_app_with_timestamp(self, msg: str) -> None:
        """
        Method called to display message on the Output Panel(upper left)
        This implementation adds the timestamp as the first element of the notification
        :param msg: The message to be notified
        """
        timestamp = pd.Timestamp.fromtimestamp(self.current_timestamp)
        self.notify_hb_app(f"({timestamp}) {msg}")

    # ----------------------------------------------------------------------------------------------------------
    # </editor-fold>
