from decimal import Decimal
from enum import Enum
from typing import Literal

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.data_types import ExecutorConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop


class DCAMode(Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"


class DCAExecutorConfig(ExecutorConfigBase):
    type: Literal["dca_executor"] = "dca_executor"
    connector_name: str
    trading_pair: str
    side: TradeType
    leverage: int = 1
    amounts_quote: list[Decimal]
    prices: list[Decimal]
    take_profit: Decimal | None = None
    stop_loss: Decimal | None = None
    trailing_stop: TrailingStop | None = None
    time_limit: int | None = None
    mode: DCAMode = DCAMode.MAKER
    activation_bounds: list[Decimal] | None = None
    level_id: str | None = None
