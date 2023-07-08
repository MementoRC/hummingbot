from dataclasses import dataclass
from decimal import Decimal

from pydantic import Field, SecretStr

import hummingbot.connector.exchange.coinbase_advanced_trade_v2.coinbase_advanced_trade_v2_constants as CONSTANTS
from hummingbot.client.config.config_data_types import BaseConnectorConfigMap, ClientFieldData
from hummingbot.core.data_type.trade_fee import TradeFeeSchema
from hummingbot.core.web_assistant.connections.data_types import EndpointRESTRequest

CENTRALIZED = True
EXAMPLE_PAIR = "ZRX-ETH"

DEFAULT_FEES = TradeFeeSchema(
    percent_fee_token="USD",
    maker_percent_fee_decimal=Decimal("0.004"),
    taker_percent_fee_decimal=Decimal("0.006"),
    buy_percent_fee_deducted_from_returns=False
)


@dataclass
class CoinbaseAdvancedTradeV2RESTRequest(EndpointRESTRequest):
    def __post_init__(self):
        super().__post_init__()
        self._ensure_endpoint_for_auth()

    @property
    def base_url(self) -> str:
        return CONSTANTS.REST_URL

    def _ensure_endpoint_for_auth(self):
        if self.is_auth_required and self.endpoint is None:
            raise ValueError("The endpoint must be specified if authentication is required.")


class CoinbaseAdvancedTradeV2ConfigMap(BaseConnectorConfigMap):
    connector: str = Field(default="coinbase_advanced_trade_v2", const=True, client_data=None)
    coinbase_advanced_trade_v2_api_key: SecretStr = Field(
        default=...,
        client_data=ClientFieldData(
            prompt=lambda cm: "Enter your Coinbase Advanced Trade API key",
            is_secure=True,
            is_connect_key=True,
            prompt_on_new=True,
        )
    )
    coinbase_advanced_trade_v2_api_secret: SecretStr = Field(
        default=...,
        client_data=ClientFieldData(
            prompt=lambda cm: "Enter your Coinbase Advanced Trade API secret",
            is_secure=True,
            is_connect_key=True,
            prompt_on_new=True,
        )
    )

    class Config:
        title = "coinbase_advanced_trade_v2"


KEYS = CoinbaseAdvancedTradeV2ConfigMap.construct()