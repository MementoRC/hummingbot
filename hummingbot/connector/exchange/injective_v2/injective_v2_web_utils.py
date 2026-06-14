from __future__ import annotations

import time

from web_assistant.throttler.async_throttler import AsyncThrottler

from hummingbot.connector.exchange.injective_v2 import injective_constants as CONSTANTS


async def get_current_server_time(
    throttler: AsyncThrottler | None = None, domain: str = CONSTANTS.DEFAULT_DOMAIN
) -> float:
    return _time() * 1e3


def _time() -> float:
    return time.time()
