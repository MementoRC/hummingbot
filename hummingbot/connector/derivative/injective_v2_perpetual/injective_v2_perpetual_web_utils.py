import time
from typing import Optional

from web_assistant.throttler.async_throttler import AsyncThrottler

from hummingbot.connector.exchange.injective_v2 import injective_constants as CONSTANTS


async def get_current_server_time(
    throttler: Optional[AsyncThrottler] = None, domain: str = CONSTANTS.DEFAULT_DOMAIN
) -> float:
    return _time() * 1e3


def _time() -> float:
    return time.time()
