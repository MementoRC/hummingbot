from decimal import Decimal
from typing import List

import pandas as pd
import pandas_ta as ta  # noqa: F401
from pydantic import Field, validator

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.progressive_trading_controller import ProgressiveTradingControllerConfig, \
    ProgressiveTradingController
from hummingbot.strategy_v2.executors.progressive_executor.data_types import ProgressiveExecutorConfig, \
    LadderedTrailingStop


class ProgressiveGainControllerConfig(ProgressiveTradingControllerConfig):
    controller_name: str = "progressive_gain"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ", )
    )
    candles_trading_pair: str = Field(
        default=None,
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ", )
    )
    interval: str = Field(
        default="30m",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=True))
    bb_length: int = Field(
        default=100,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Bollinger Bands length: ",
            prompt_on_new=True))
    bb_std: float = Field(
        default=2.0,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the Bollinger Bands standard deviation: ",
            prompt_on_new=False))
    bb_long_threshold: float = Field(
        default=0.0,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter the Bollinger Bands long threshold: ",
            prompt_on_new=True))
    bb_short_threshold: float = Field(
        default=1.0,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda mi: "Enter the Bollinger Bands short threshold: ",
            prompt_on_new=True))
    macd_fast: int = Field(
        default=21,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD fast period: ",
            prompt_on_new=True))
    macd_slow: int = Field(
        default=42,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD slow period: ",
            prompt_on_new=True))
    macd_signal: int = Field(
        default=9,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the MACD signal period: ",
            prompt_on_new=True))
    dynamic_order_spread: bool = Field(
        default=True,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enable dynamic order spread: ",
            prompt_on_new=True))
    dynamic_target: bool = Field(
        default=True,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enable dynamic target: ",
            prompt_on_new=True))

    @validator("candles_connector", pre=True, always=True)
    def set_candles_connector(cls, v, values):
        return values.get("connector_name") if v is None or v == "" else v

    @validator("candles_trading_pair", pre=True, always=True)
    def set_candles_trading_pair(cls, v, values):
        return values.get("trading_pair") if v is None or v == "" else v


class ProgressiveGainController(ProgressiveTradingController):
    """
    Mean reversion strategy with Grid execution making use of Bollinger Bands indicator to make spreads dynamic
    and shift the mid-price.
    """

    def __init__(self, config: ProgressiveGainControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = config.bb_length
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

        self._volatility: float = 0.0

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(connector_name=self.config.candles_connector,
                                                      trading_pair=self.config.candles_trading_pair,
                                                      interval=self.config.interval,
                                                      max_records=self.max_records)
        # Add indicators
        df.ta.bbands(length=self.config.bb_length, std=self.config.bb_std, append=True)
        df.ta.natr(length=self.config.bb_length)
        df.ta.macd(fast=self.config.macd_fast, slow=self.config.macd_slow, signal=self.config.macd_signal, append=True)
        df.ta.adx(length=self.config.bb_length, append=True)
        df.ta.aroon(length=self.config.macd_fast // 2, append=True)
        df.ta.aroon(length=self.config.macd_fast, append=True)
        df.ta.aroon(length=self.config.macd_fast * 2, append=True)
        df.ta.aroon(length=self.config.macd_fast * 4, append=True)

        # bbp = df[f"BBP_{self.config.bb_length}_{self.config.bb_std}"]
        # macdh = df[f"MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
        macd = df[f"MACD_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
        macds = df[f"MACDs_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
        aroon_0 = df[f"AROONOSC_{self.config.macd_fast // 2}"]
        aroon_1 = df[f"AROONOSC_{self.config.macd_fast}"]
        aroon_2 = df[f"AROONOSC_{self.config.macd_fast * 2}"]
        # aroon_3 = df[f"AROONOSC_{self.config.macd_fast * 4}"]

        # Add condition for macdh > macds
        df["MACD>S"] = 0
        df.loc[macd > macds, "MACD>S"] = 1
        df["MACD_cross"] = df["MACD>S"].diff()

        # Generate signal
        long_condition = (
            (df["MACD>S"] == 1) &
            (aroon_0 > 0) &
            (aroon_1 > 0) &
            (aroon_2 > 0)  # &
            # (aroon_3 > 0)
        )
        short_condition = (
            (df["MACD>S"] == 0) &
            (aroon_0 < 0) &
            (aroon_1 < 0) &
            (aroon_2 < 0)  # &
            # (aroon_3 < 0)
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
        df["signal"] = 0
        df.loc[long_condition, "signal"] = -1
        df.loc[short_condition, "signal"] = 1

        df["volatility"] = df[f"BBB_{self.config.bb_length}_{self.config.bb_std}"] / self.config.bb_std / 100
        if df["volatility"].iloc[-1] != 0:
            volatility_update = (
                abs((df["volatility"].iloc[-1] - self._volatility) / df["volatility"].iloc[-1]) > 0.01
            )
            self._volatility = df["volatility"].iloc[-1]
        else:
            volatility_update = False

        def format_to_4g(decimal_value: Decimal) -> Decimal:
            return Decimal(f"{decimal_value:.4g}")

        for col in df.columns:
            if df[col].dtype == "float64":
                df[col] = df[col].apply(format_to_4g)

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["volatility_update"] = volatility_update
        self.processed_data["volatility"] = df["volatility"].iloc[-1]

        if self.processed_data['volatility_update']:
            self.logger().info(f"Progressive Gain Volatility: {self.processed_data['volatility']}")
            self.logger().info(f"Progressive Gain Volatility update: {self.processed_data['volatility_update']}")

        self.processed_data["features"] = df[
            [
                "timestamp", "open", "high", "low", "close", "volume",
                f"BBP_{self.config.bb_length}_{self.config.bb_std}",
                "MACD>S",
                f"AROONOSC_{self.config.macd_fast // 2}",
                f"AROONOSC_{self.config.macd_fast}",
                f"AROONOSC_{self.config.macd_fast * 2}",
                # f"AROONOSC_{self.config.macd_fast * 4}",
                "signal",
            ]
        ]
        # await asyncio.sleep(0)

    def get_spread_multiplier(self) -> Decimal:
        if self.config.dynamic_order_spread:
            df = self.processed_data["features"]
            bb_width = df[f"BBB_{self.config.bb_length}_{self.config.bb_std}"].iloc[-1]
            return Decimal(bb_width / 200)
        else:
            return Decimal("1.0")

    def get_executor_config(self, trade_type: TradeType, price: Decimal, amount: Decimal) -> ProgressiveExecutorConfig:
        spread_multiplier = self.get_spread_multiplier()
        if self.config.dynamic_target:
            stop_loss = self.config.stop_loss * spread_multiplier
            trailing_stop = LadderedTrailingStop(
                activation_pnl_pct=self.config.trailing_stop.activation_pnl_pct * spread_multiplier,
                trailing_pct=self.config.trailing_stop.trailing_pct * spread_multiplier,
                take_profit_table=self.config.trailing_stop.take_profit_table
            )
        else:
            stop_loss = self.config.stop_loss
            trailing_stop = self.config.trailing_stop

        self.config.triple_barrier_config.stop_loss = stop_loss
        self.config.triple_barrier_config.trailing_stop = trailing_stop
        return ProgressiveExecutorConfig(
            type="progressive_executor",
            timestamp=self.market_data_provider.time(),
            trading_pair=self.config.trading_pair,
            connector_name=self.config.connector_name,
            side=trade_type,
            entry_price=price,
            amount=amount,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage,
        )
