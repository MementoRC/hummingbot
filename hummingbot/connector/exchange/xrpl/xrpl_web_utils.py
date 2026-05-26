import time

from web_assistant.throttler.async_throttler import AsyncThrottler

import hummingbot.connector.exchange.cube.cube_constants as CONSTANTS


async def get_current_server_time(
    throttler: AsyncThrottler | None = None,
    domain: str = CONSTANTS.DEFAULT_DOMAIN,
) -> float:
    return time.time()
