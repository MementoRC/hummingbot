from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from market_connector.exchanges.kraken.linked_order_codec import KrakenClOrdIdLinkCodec
from market_connector.hb_compat.registry import get_exchange

from hummingbot.connector.market_connector.kraken_emulation_venue import KrakenEmulationVenue
from hummingbot.connector.market_connector.market_connector_exchange import MarketConnectorExchange

CONNECTOR_NAME = "kraken_mc"
# Seconds an UNKNOWN (lost-create) order may stay unresolved before the core fails it (plan D3).
UNKNOWN_ORDER_GRACE_S = 60.0


class KrakenMcExchange(MarketConnectorExchange):
    """Kraken through market-connector (temporary name, hummingbot#85)."""

    def __init__(
        self,
        kraken_mc_api_key: str,
        kraken_mc_secret_key: str,
        kraken_mc_api_tier: str = "Starter",
        balance_asset_limit: Optional[Dict[str, Dict[str, Decimal]]] = None,
        rate_limits_share_pct: Decimal = Decimal("100"),
        trading_pairs: Optional[List[str]] = None,
        trading_required: bool = True,
        order_book_refresh_interval: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(
            descriptor=get_exchange("kraken"),
            config={
                "kraken_api_key": kraken_mc_api_key,
                "kraken_secret_key": kraken_mc_secret_key,
                "kraken_api_tier": kraken_mc_api_tier,
            },
            connector_name=CONNECTOR_NAME,
            trading_pairs=trading_pairs,
            trading_required=trading_required,
            balance_asset_limit=balance_asset_limit,
            rate_limits_share_pct=rate_limits_share_pct,
            core_options={"unknown_order_grace": UNKNOWN_ORDER_GRACE_S, "own_trades_ws": True},
            order_book_refresh_interval=order_book_refresh_interval,
        )

    def _build_linked_venue(self, handle: Any) -> Tuple[Any, Any]:
        return KrakenEmulationVenue(handle.gateway), KrakenClOrdIdLinkCodec()
