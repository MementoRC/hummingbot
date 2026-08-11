from typing import Literal

from hummingbot.core.data_type.common import OrderType
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig


class PositionOnExchangeExecutorConfig(PositionExecutorConfig):
    type: Literal["position_on_exchange_executor"] = "position_on_exchange_executor"
    triple_barrier_config: TripleBarrierConfig = TripleBarrierConfig(
        stop_loss=None,
        take_profit=None,
        stop_loss_order_type=OrderType.STOP_LOSS,
    )
