from decimal import Decimal

from pydantic import Field, SecretStr

from data_type_primitives.trade_fee import TradeFeeSchema
from hummingbot.client.config.config_data_types import BaseConnectorConfigMap

DEFAULT_FEES = TradeFeeSchema(
    maker_percent_fee_decimal=Decimal("0.0002"),
    taker_percent_fee_decimal=Decimal("0.0006"),
    buy_percent_fee_deducted_from_returns=True,
)

CENTRALIZED = True

EXAMPLE_PAIR = "BTC-USDT"

BROKER_ID = "x-3QreWesy"


class BitmartPerpetualConfigMap(BaseConnectorConfigMap):
    connector: str = "bitmart_perpetual"
    bitmart_perpetual_api_key: SecretStr = Field(
        default=...,
        json_schema_extra={
            "prompt": "Enter your Bitmart Perpetual API key",
            "is_secure": True,
            "is_connect_key": True,
            "prompt_on_new": True,
        },
    )
    bitmart_perpetual_api_secret: SecretStr = Field(
        default=...,
        json_schema_extra={
            "prompt": "Enter your Bitmart Perpetual API secret",
            "is_secure": True,
            "is_connect_key": True,
            "prompt_on_new": True,
        },
    )
    bitmart_perpetual_memo: SecretStr = Field(
        default=...,
        json_schema_extra={
            "prompt": "Enter your Bitmart Perpetual Memo",
            "is_secure": True,
            "is_connect_key": True,
            "prompt_on_new": True,
        },
    )


KEYS = BitmartPerpetualConfigMap.model_construct()
