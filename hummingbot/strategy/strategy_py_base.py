from hummingbot.core.clock import Clock
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    FundingPaymentCompletedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    PositionModeChangeEvent,
    RangePositionLiquidityAddedEvent,
    RangePositionLiquidityRemovedEvent,
    RangePositionUpdateFailureEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.strategy.strategy_base import StrategyBase


class StrategyPyBase(StrategyBase):
    def __init__(self):
        super().__init__()

    def c_start(self, clock: Clock, timestamp: float):
        StrategyBase.c_start(self, clock, timestamp)
        self.start(clock, timestamp)

    def start(self, clock: Clock, timestamp: float):
        pass

    def c_stop(self, clock: Clock):
        StrategyBase.c_stop(self, clock)
        self.stop(clock)

    def stop(self, clock: Clock):
        pass

    def c_tick(self, timestamp: float):
        StrategyBase.c_tick(self, timestamp)
        self.tick(timestamp)

    def tick(self, timestamp: float):
        raise NotImplementedError

    def c_did_create_buy_order(self, order_created_event: object):
        self.did_create_buy_order(order_created_event)

    def did_create_buy_order(self, order_created_event: BuyOrderCreatedEvent):
        pass

    def c_did_create_sell_order(self, order_created_event: object):
        self.did_create_sell_order(order_created_event)

    def did_create_sell_order(self, order_created_event: SellOrderCreatedEvent):
        pass

    def c_did_fill_order(self, order_filled_event: object):
        self.did_fill_order(order_filled_event)

    def did_fill_order(self, order_filled_event: OrderFilledEvent):
        pass

    def c_did_fail_order(self, order_failed_event: object):
        self.did_fail_order(order_failed_event)

    def did_fail_order(self, order_failed_event: MarketOrderFailureEvent):
        pass

    def c_did_cancel_order(self, cancelled_event: object):
        self.did_cancel_order(cancelled_event)

    def did_cancel_order(self, cancelled_event: OrderCancelledEvent):
        pass

    def c_did_expire_order(self, expired_event: object):
        self.did_expire_order(expired_event)

    def did_expire_order(self, expired_event: OrderExpiredEvent):
        pass

    def c_did_complete_buy_order(self, order_completed_event: object):
        self.did_complete_buy_order(order_completed_event)

    def did_complete_buy_order(self, order_completed_event: BuyOrderCompletedEvent):
        pass

    def c_did_complete_sell_order(self, order_completed_event: object):
        self.did_complete_sell_order(order_completed_event)

    def did_complete_sell_order(self, order_completed_event: SellOrderCompletedEvent):
        pass

    def c_did_complete_funding_payment(self, funding_payment_completed_event: object):
        self.did_complete_funding_payment(funding_payment_completed_event)

    def did_complete_funding_payment(self, funding_payment_completed_event: FundingPaymentCompletedEvent):
        pass

    def c_did_change_position_mode_succeed(self, position_mode_changed_event: object):
        self.did_change_position_mode_succeed(position_mode_changed_event)

    def did_change_position_mode_succeed(self, position_mode_changed_event: PositionModeChangeEvent):
        pass

    def c_did_change_position_mode_fail(self, position_mode_changed_event: object):
        self.did_change_position_mode_fail(position_mode_changed_event)

    def did_change_position_mode_fail(self, position_mode_changed_event: PositionModeChangeEvent):
        pass

    def c_did_add_liquidity(self, add_liquidity_event: object):
        self.did_add_liquidity(add_liquidity_event)

    def did_add_liquidity(self, add_liquidity_event: RangePositionLiquidityAddedEvent):
        pass

    def c_did_remove_liquidity(self, remove_liquidity_event: object):
        self.did_remove_liquidity(remove_liquidity_event)

    def did_remove_liquidity(self, remove_liquidity_event: RangePositionLiquidityRemovedEvent):
        pass

    def c_did_fail_lp_update(self, fail_lp_update_event: object):
        self.did_fail_lp_update(fail_lp_update_event)

    def did_fail_lp_update(self, fail_lp_update_event: RangePositionUpdateFailureEvent):
        pass
