import time

from hummingbot.core.api_throttler.async_throttler import AsyncThrottler


async def get_current_server_time(
    throttler: AsyncThrottler | None = None,
    domain: str = "",
) -> float:
    return time.time()
