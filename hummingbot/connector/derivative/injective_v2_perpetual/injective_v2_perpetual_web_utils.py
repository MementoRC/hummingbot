import time

from hummingbot.connector.exchange.injective_v2 import injective_constants as CONSTANTS
from hummingbot.core.api_throttler.async_throttler import AsyncThrottler


async def get_current_server_time(
    throttler: AsyncThrottler | None = None, domain: str = CONSTANTS.DEFAULT_DOMAIN
) -> float:
    return _time() * 1e3


def _time() -> float:
    return time.time()
