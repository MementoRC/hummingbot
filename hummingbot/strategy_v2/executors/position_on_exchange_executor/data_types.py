from typing import Literal

from pydantic import model_validator

from hummingbot.core.data_type.common import OrderType
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.executors.validation import require_market_order_type, require_positive


class PositionOnExchangeTripleBarrierConfig(TripleBarrierConfig):
    """
    Triple barrier config for exchange-native stop-loss placement.

    ``TripleBarrierConfig.validate_barriers`` requires ``stop_loss_order_type`` to be
    MARKET because the base ``PositionExecutor`` monitors the stop client-side and needs
    an order type that closes the position immediately at any price once its condition is
    detected. ``PositionOnExchangeExecutor`` instead registers the stop directly on the
    exchange as a STOP_LOSS or STOP_LOSS_LIMIT order, so the exchange itself triggers the
    close at the stop price and the base MARKET-only requirement does not apply. Every
    other barrier check is unchanged.
    """

    @model_validator(mode="after")
    def validate_barriers(self):
        # The barriers are distances from the entry price, so a non positive value would
        # place the barrier at or beyond the entry and close the position immediately.
        require_positive("triple_barrier_config.stop_loss", self.stop_loss)
        require_positive("triple_barrier_config.take_profit", self.take_profit)
        require_positive("triple_barrier_config.time_limit", self.time_limit)
        # Time limit has to close the position at any price, so it can't be a resting limit order.
        require_market_order_type("triple_barrier_config.time_limit_order_type", self.time_limit_order_type)
        # The stop is registered on the exchange, so it must be one of the exchange-native
        # stop order types instead of the client-side-only MARKET requirement.
        if self.stop_loss_order_type not in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT):
            raise ValueError(
                f"triple_barrier_config.stop_loss_order_type ({self.stop_loss_order_type.name}) "
                f"must be STOP_LOSS or STOP_LOSS_LIMIT"
            )
        return self


class PositionOnExchangeExecutorConfig(PositionExecutorConfig):
    type: Literal["position_on_exchange_executor"] = "position_on_exchange_executor"
    triple_barrier_config: PositionOnExchangeTripleBarrierConfig = PositionOnExchangeTripleBarrierConfig(
        stop_loss=None,
        take_profit=None,
        stop_loss_order_type=OrderType.STOP_LOSS,
    )
