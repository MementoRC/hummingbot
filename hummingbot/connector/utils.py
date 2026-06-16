"""Connector utilities — shim re-exporting pure helpers from connector_utils.

The 5 trading-pair / order-id pure helpers live in the standalone
connector_utils sub-package (sub-packages/connector-utils). This module:
- Re-exports the 3 trading-pair helpers directly (signatures identical).
- Wraps the 2 client-order-id factories to inject hummingbot's tracking_nonce
  via the new nonce= kwarg, preserving the original signatures.
- Keeps TimeSynchronizerRESTPreProcessor + GZipCompressionWSPostProcessor
  INLINE (they need hummingbot-specific TimeSynchronizer + web_assistant types
  — not portable to the sub-package).
- Preserves to_0x_hex, build_api_factory, and TradeFillOrderDetails inline
  (out of extraction scope).
"""

from __future__ import annotations

import gzip
import json
from collections import namedtuple
from typing import Any, Callable

# Sub-package imports below. Installed via `pixi run install-subpackages` (see pyproject.toml).
from async_utils.tracking_nonce import NonceCreator, get_tracking_nonce
from connector_utils import (
    combine_to_hb_trading_pair,  # noqa: F401  — re-export for callers
    split_hb_trading_pair,  # noqa: F401  — re-export for callers
    validate_trading_pair,  # noqa: F401  — re-export for callers
)
from connector_utils.client_order_id import (
    get_new_client_order_id as _gen_client_order_id,
    get_new_numeric_client_order_id as _gen_numeric_client_order_id,
)
from hexbytes import HexBytes
from web_assistant.connections.data_types import RESTRequest, WSResponse
from web_assistant.rest_pre_processors import RESTPreProcessorBase
from web_assistant.throttler.async_throttler import AsyncThrottler
from web_assistant.throttler.async_throttler_base import AsyncThrottlerBase
from web_assistant.web_assistants_factory import WebAssistantsFactory
from web_assistant.ws_post_processors import WSPostProcessorBase

from hummingbot.connector.time_synchronizer import TimeSynchronizer

TradeFillOrderDetails = namedtuple("TradeFillOrderDetails", "market exchange_trade_id symbol")


def build_api_factory(throttler: AsyncThrottlerBase) -> WebAssistantsFactory:
    throttler = throttler or AsyncThrottler(rate_limits=[])
    api_factory = WebAssistantsFactory(throttler=throttler)
    return api_factory


def get_new_client_order_id(
    is_buy: bool,
    trading_pair: str,
    hbot_order_id_prefix: str = "",
    max_id_len: int | None = None,
) -> str:
    """Generate a unique client order ID using hummingbot's tracking_nonce.

    Delegates to connector_utils.client_order_id.get_new_client_order_id with
    nonce injected from async_utils.tracking_nonce.get_tracking_nonce().
    """
    return _gen_client_order_id(
        is_buy=is_buy,
        trading_pair=trading_pair,
        hbot_order_id_prefix=hbot_order_id_prefix,
        max_id_len=max_id_len,
        nonce=get_tracking_nonce(),
    )


def get_new_numeric_client_order_id(nonce_creator: NonceCreator, max_id_bit_count: int | None = None) -> int:
    """Generate a unique numeric client order ID using NonceCreator.

    Delegates to connector_utils.client_order_id.get_new_numeric_client_order_id
    with nonce from the supplied NonceCreator. Preserves the original signature
    so inbound callers don't need changes.
    """
    return _gen_numeric_client_order_id(
        max_id_bit_count=max_id_bit_count,
        nonce=nonce_creator.get_tracking_nonce(),
    )


class TimeSynchronizerRESTPreProcessor(RESTPreProcessorBase):
    """
    This pre processor is intended to be used in those connectors that require synchronization with the server time
    to accept API requests. It ensures the synchronizer has at least one server time sample before being used.
    """

    def __init__(self, synchronizer: TimeSynchronizer, time_provider: Callable):
        super().__init__()
        self._synchronizer = synchronizer
        self._time_provider = time_provider

    async def pre_process(self, request: RESTRequest) -> RESTRequest:
        await self._synchronizer.update_server_time_if_not_initialized(time_provider=self._time_provider())
        return request


class GZipCompressionWSPostProcessor(WSPostProcessorBase):
    """
    Performs the necessary response processing from both public and private websocket streams.
    """

    async def post_process(self, response: WSResponse) -> WSResponse:
        if not isinstance(response.data, bytes):
            # Unlike Market WebSocket, the return data of Account and Order Websocket are not compressed by GZIP.
            return response
        encoded_msg: bytes = gzip.decompress(response.data)
        msg: dict[str, Any] = json.loads(encoded_msg.decode("utf-8"))

        return WSResponse(data=msg)


def to_0x_hex(signature: HexBytes | bytes) -> str:
    """
    Convert a string to a 0x-prefixed hex string.
    """
    if hasattr(signature, "to_0x_hex"):
        return signature.to_0x_hex()

    return hex if (hex := signature.hex()).startswith("0x") else f"0x{hex}"
