from decimal import Decimal

import pandas as pd
from pydantic import Field, validator

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.controllers import DirectionalTradingControllerConfigBase, DirectionalTradingControllerBase
from hummingbot.strategy_v2.executors.progressive_executor.data_types import LadderedTrailingStop, \
    YieldTripleBarrierConfig, ProgressiveExecutorUpdates, ProgressiveExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, UpdateExecutorAction


class ProgressiveTradingControllerConfig(DirectionalTradingControllerConfigBase):
    """
    This class represents the configuration required to run a Progressive Strategy.
    It adds the concept of APR yield to the Directional Strategy.
    It modifies the Triple Barrier Configuration to include a ladder of trailing stops.
    """
    controller_type = "progressive_trading"
    # Triple Barrier Configuration
    apr_yield: Decimal | None = Field(
        default=Decimal("0.5"), gt=0,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter the APR yield (as a decimal, e.g., 0.5 for 50%): ",
            prompt_on_new=True))
    trailing_stop: LadderedTrailingStop | None = Field(
        default="0.015,0.005,0.05:1|0.1:0.91|0.25:0.8|0.5:0.5",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the trailing stop as activation_pnl_pct,trailing_pct,profit_table (e.g., 0.015,0.003,0.05:1|0.1:0.91): ",
            prompt_on_new=True))

    @validator("trailing_stop", pre=True, always=True)
    def parse_trailing_stop(cls, v):
        if isinstance(v, str):
            if v == "":
                return None
            activation_pnl_pct, trailing_pct, *take_profit_table = v.split(",")
            take_profit_table = tuple(map(lambda x: tuple(map(Decimal, x.split(":"))), take_profit_table[0].split("|")))
            return LadderedTrailingStop(
                activation_pnl_pct=Decimal(activation_pnl_pct),
                trailing_pct=Decimal(trailing_pct),
                take_profit_table=take_profit_table
            )
        return v

    @validator("apr_yield", pre=True, always=True)
    def validate_target(cls, v):
        if isinstance(v, str):
            return None if v == "" else Decimal(v)
        return v

    @property
    def triple_barrier_config(self) -> YieldTripleBarrierConfig:
        return YieldTripleBarrierConfig(
            apr_yield=self.apr_yield,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            time_limit=self.time_limit,
            trailing_stop=self.trailing_stop,
            open_order_type=OrderType.MARKET,  # Defaulting to MARKET as is a Taker Controller
            take_profit_order_type=self.take_profit_order_type,
            stop_loss_order_type=OrderType.MARKET,  # Defaulting to MARKET as per requirement
            time_limit_order_type=OrderType.MARKET  # Defaulting to MARKET as per requirement
        )


class ProgressiveTradingController(DirectionalTradingControllerBase):
    """
    This class represents the base class for a Directional Strategy.
    """
    def __init__(self, config: ProgressiveTradingControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def determine_executor_actions(self) -> list[ExecutorAction]:
        """
        Determine actions based on the provided executor handler report.
        """
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.update_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions

    def update_actions_proposal(self) -> list[ExecutorAction]:
        """
        Stop actions based on the provided executor handler report.
        """
        update_actions = []
        if self.processed_data.get("volatility_update", 0) != 0:
            self.logger().info(f"Volatility actions proposal: {self.processed_data['volatility']}")
            update_actions.extend(self.executors_to_update())
        return update_actions

    def executors_to_update(self) -> list[ExecutorAction]:
        executors_to_refresh = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_trading)

        return [
            UpdateExecutorAction(
                controller_id=self.config.id,
                executor_id=executor.id,
                update_data=ProgressiveExecutorUpdates(volatility=self.processed_data["volatility"])
            ) for executor in executors_to_refresh
        ]

    def get_executor_config(self, trade_type: TradeType, price: Decimal, amount: Decimal) -> ProgressiveExecutorConfig:
        """
        Get the executor config based on the trade_type, price and amount. This method can be overridden by the
        subclasses if required.
        """
        return ProgressiveExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=trade_type,
            entry_price=price,
            amount=amount,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage,
        )

    def to_format_status(self) -> list[str]:
        df = self.processed_data.get("features", pd.DataFrame())
        if df.empty:
            return []
        return [format_df_for_printout(df.tail(1), table_format="psql",)]
