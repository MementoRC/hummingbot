from __future__ import annotations

import time

from web_assistant.web_assistants_factory import WebAssistantsFactory

from hummingbot.connector.derivative.pacifica_perpetual import pacifica_perpetual_constants as CONSTANTS
from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.web_assistant.auth import AuthBase


def public_rest_url(path_url: str, domain: str = CONSTANTS.DEFAULT_DOMAIN) -> str:
    base_url = CONSTANTS.REST_URL if domain == CONSTANTS.DEFAULT_DOMAIN else CONSTANTS.TESTNET_REST_URL
    return base_url + path_url


def private_rest_url(path_url: str, domain: str = CONSTANTS.DEFAULT_DOMAIN) -> str:
    return public_rest_url(path_url, domain)


def wss_url(domain: str = CONSTANTS.DEFAULT_DOMAIN) -> str:
    return CONSTANTS.WSS_URL if domain == CONSTANTS.DEFAULT_DOMAIN else CONSTANTS.TESTNET_WSS_URL


def build_api_factory(
    throttler: AsyncThrottler | None = None,
    auth: AuthBase | None = None,
) -> WebAssistantsFactory:
    throttler = throttler or AsyncThrottler(CONSTANTS.RATE_LIMITS)
    api_factory = WebAssistantsFactory(
        throttler=throttler,
        auth=auth,
    )
    return api_factory


async def get_current_server_time(
    throttler: AsyncThrottler | None = None,
    domain: str = CONSTANTS.DEFAULT_DOMAIN,
) -> float:
    return time.time()
