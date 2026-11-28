from decimal import Decimal
from typing import Optional

from pydantic import ConfigDict, Field, SecretStr, field_validator

from hummingbot.client.config.config_data_types import BaseConnectorConfigMap
from hummingbot.core.data_type.trade_fee import TradeFeeSchema

CENTRALIZED = True

EXAMPLE_PAIR = "ETH-USDC"

DEFAULT_FEES = TradeFeeSchema(
    maker_percent_fee_decimal=Decimal("0.0025"),
    taker_percent_fee_decimal=Decimal("0.004"),
)

API_TIERS = ("STARTER", "INTERMEDIATE", "PRO")


class KrakenMcConfigMap(BaseConnectorConfigMap):
    connector: str = "kraken_mc"
    kraken_mc_api_key: SecretStr = Field(
        default=...,
        json_schema_extra={
            "prompt": "Enter your Kraken API key",
            "is_secure": True,
            "is_connect_key": True,
            "prompt_on_new": True,
        },
    )
    kraken_mc_secret_key: SecretStr = Field(
        default=...,
        json_schema_extra={
            "prompt": "Enter your Kraken secret key",
            "is_secure": True,
            "is_connect_key": True,
            "prompt_on_new": True,
        },
    )
    kraken_mc_api_tier: str = Field(
        default="Starter",
        json_schema_extra={
            "prompt": "Enter your Kraken API Tier (Starter/Intermediate/Pro)",
            "prompt_on_new": True,
        },
    )
    kraken_mc_host_slot: int = Field(
        default=0,
        ge=0,
        le=99,
        json_schema_extra={
            "prompt": "Enter this host's linked-order slot (0-99, unique per host on the same Kraken account)",
            "prompt_on_new": True,
        },
    )
    model_config = ConfigDict(title="kraken_mc")

    @field_validator("kraken_mc_api_tier", mode="before")
    @classmethod
    def _api_tier_validator(cls, value: str) -> Optional[str]:
        if str(value).upper() not in API_TIERS:
            raise ValueError("No such Kraken API Tier.")
        return value


KEYS = KrakenMcConfigMap.model_construct()
