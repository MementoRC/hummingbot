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

    def did_create_buy_order(self, order_created_event: BuyOrderCreatedEvent):
        pass

    def did_create_sell_order(self, order_created_event: SellOrderCreatedEvent):
        pass

    def did_fill_order(self, order_filled_event: OrderFilledEvent):
        pass

    def did_fail_order(self, order_failed_event: MarketOrderFailureEvent):
        pass

    def did_cancel_order(self, cancelled_event: OrderCancelledEvent):
        pass

    def did_expire_order(self, expired_event: OrderExpiredEvent):
        pass

    def did_complete_buy_order(self, order_completed_event: BuyOrderCompletedEvent):
        pass

    def did_complete_sell_order(self, order_completed_event: SellOrderCompletedEvent):
        pass

    def did_complete_funding_payment(self, funding_payment_completed_event: FundingPaymentCompletedEvent):
        pass

    def did_change_position_mode_succeed(self, position_mode_changed_event: PositionModeChangeEvent):
        pass

    def did_change_position_mode_fail(self, position_mode_changed_event: PositionModeChangeEvent):
        pass

    def did_add_liquidity(self, add_liquidity_event: RangePositionLiquidityAddedEvent):
        pass

    def did_remove_liquidity(self, remove_liquidity_event: RangePositionLiquidityRemovedEvent):
        pass

    def did_fail_lp_update(self, fail_lp_update_event: RangePositionUpdateFailureEvent):
        pass
