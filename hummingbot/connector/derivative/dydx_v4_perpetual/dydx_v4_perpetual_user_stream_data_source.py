from __future__ import annotations

import asyncio
import logging
from typing import Any

from web_assistant.web_assistants_factory import WebAssistantsFactory

import hummingbot.connector.derivative.dydx_v4_perpetual.dydx_v4_perpetual_constants as CONSTANTS
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.web_assistant.connections.data_types import WSJSONRequest
from hummingbot.core.web_assistant.ws_assistant import WSAssistant
from hummingbot.logger import HummingbotLogger


class DydxV4PerpetualUserStreamDataSource(UserStreamTrackerDataSource):
    _logger: HummingbotLogger | None = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(self, api_factory: WebAssistantsFactory | None, connector):
        self._api_factory: WebAssistantsFactory = api_factory
        self._ws_assistant: WSAssistant | None = None
        self._connector = connector

        super().__init__()

    @property
    def last_recv_time(self):
        if self._ws_assistant:
            return self._ws_assistant.last_recv_time
        return -1

    async def _subscribe_channels(self, websocket_assistant: WSAssistant):
        pass

    # ping的回调还没写，不然会断掉
    async def _connected_websocket_assistant(self) -> WSAssistant:
        if self._ws_assistant is None:
            self.logger().info(f"Connecting to {CONSTANTS.DYDX_V4_WS_URL}")
            self._ws_assistant = await self._api_factory.get_ws_assistant()
            await self._ws_assistant.connect(ws_url=CONSTANTS.DYDX_V4_WS_URL, ping_timeout=CONSTANTS.HEARTBEAT_INTERVAL)

            subaccount_id = f"{self._connector._dydx_v4_perpetual_chain_address}/{self._connector.subaccount_id}"

            subscribe_account_request: WSJSONRequest = WSJSONRequest(
                payload={
                    "type": "subscribe",
                    "channel": CONSTANTS.WS_CHANNEL_ACCOUNTS,
                    "id": subaccount_id,
                },
                is_auth_required=False,
            )
            await self._ws_assistant.send(subscribe_account_request)
            self.logger().info("Authenticated user stream...")
        return self._ws_assistant

    async def _process_event_message(self, event_message: dict[str, Any], queue: asyncio.Queue):
        if event_message.get("type", "") in [CONSTANTS.WS_TYPE_SUBSCRIBED, CONSTANTS.WS_TYPE_CHANNEL_DATA]:
            await super()._process_event_message(event_message=event_message, queue=queue)
