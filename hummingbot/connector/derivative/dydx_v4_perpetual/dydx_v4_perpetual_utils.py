from decimal import Decimal

from pydantic import ConfigDict, Field, SecretStr

from data_type_primitives.trade_fee import TradeFeeSchema
from hummingbot.client.config.config_data_types import BaseConnectorConfigMap

CENTRALIZED = True

EXAMPLE_PAIR = "BTC-USD"

DEFAULT_FEES = TradeFeeSchema(
    maker_percent_fee_decimal=Decimal("0.0001"),
    taker_percent_fee_decimal=Decimal("0.0005"),
)


def clamp(value, minvalue, maxvalue):
    return max(minvalue, min(value, maxvalue))


class DydxV4PerpetualConfigMap(BaseConnectorConfigMap):
    connector: str = "dydx_v4_perpetual"
    dydx_v4_perpetual_secret_phrase: SecretStr = Field(
        default=...,
        json_schema_extra={
            "prompt": "Enter your dydx v4 secret_phrase(24 words)",
            "is_secure": True,
            "is_connect_key": True,
            "prompt_on_new": True,
        },
    )
    dydx_v4_perpetual_chain_address: SecretStr = Field(
        default=...,
        json_schema_extra={
            "prompt": "Enter your dydx v4 chain address ( starts with 'dydx' )",
            "is_secure": True,
            "is_connect_key": True,
            "prompt_on_new": True,
        },
    )
    model_config = ConfigDict(title="dydx_v4_perpetual")


KEYS = DydxV4PerpetualConfigMap.model_construct()
