"""Compat shim: hummingbot.core.api_throttler -> web_assistant.throttler."""

import sys

import web_assistant.throttler.async_request_context_base as _async_request_context_base
import web_assistant.throttler.async_throttler as _async_throttler
import web_assistant.throttler.async_throttler_base as _async_throttler_base
import web_assistant.throttler.data_types as _data_types

sys.modules["hummingbot.core.api_throttler.async_request_context_base"] = _async_request_context_base
sys.modules["hummingbot.core.api_throttler.async_throttler"] = _async_throttler
sys.modules["hummingbot.core.api_throttler.async_throttler_base"] = _async_throttler_base
sys.modules["hummingbot.core.api_throttler.data_types"] = _data_types

from web_assistant.throttler.async_request_context_base import (  # noqa: E402
    MAX_CAPACITY_REACHED_WARNING_INTERVAL,
    AsyncRequestContextBase,
)
from web_assistant.throttler.async_throttler import AsyncRequestContext, AsyncThrottler  # noqa: E402
from web_assistant.throttler.async_throttler_base import AsyncThrottlerBase  # noqa: E402
from web_assistant.throttler.data_types import (  # noqa: E402
    DEFAULT_PATH,
    DEFAULT_WEIGHT,
    Limit,
    LinkedLimitWeightPair,
    RateLimit,
    RequestPath,
    RequestWeight,
    Seconds,
    TaskLog,
)

__all__ = [
    "MAX_CAPACITY_REACHED_WARNING_INTERVAL",
    "AsyncRequestContextBase",
    "AsyncRequestContext",
    "AsyncThrottler",
    "AsyncThrottlerBase",
    "DEFAULT_PATH",
    "DEFAULT_WEIGHT",
    "LinkedLimitWeightPair",
    "Limit",
    "RateLimit",
    "RequestPath",
    "RequestWeight",
    "Seconds",
    "TaskLog",
]
