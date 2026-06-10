"""Pure-Python port of api_asset_price_delegate.pyx.

APIAssetPriceDelegate fetches mid-price from a custom HTTP endpoint
via CustomAPIDataFeed instead of reading an order book.
"""

from __future__ import annotations

from decimal import Decimal

from hummingbot.core.data_type.common import PriceType
from hummingbot.data_feed.custom_api_data_feed import CustomAPIDataFeed, NetworkStatus
from hummingbot.strategy.asset_price_delegate import AssetPriceDelegate


class APIAssetPriceDelegate(AssetPriceDelegate):
    def __init__(self, market: object, api_url: str, update_interval: float = 5.0) -> None:
        super().__init__()
        self._market = market
        self._custom_api_feed = CustomAPIDataFeed(api_url=api_url, update_interval=update_interval)
        self._custom_api_feed.start()

    def get_price_by_type(self, _: PriceType) -> Decimal:
        return self.c_get_mid_price()

    def c_get_mid_price(self) -> Decimal:
        return self._custom_api_feed.get_price()

    @property
    def ready(self) -> bool:
        return self._custom_api_feed.network_status == NetworkStatus.CONNECTED

    @property
    def market(self) -> object:
        return self._market

    @property
    def custom_api_feed(self) -> CustomAPIDataFeed:
        return self._custom_api_feed
