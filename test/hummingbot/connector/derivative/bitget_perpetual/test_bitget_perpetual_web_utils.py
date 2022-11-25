import asyncio
import json
import unittest
from typing import Awaitable

from aioresponses import aioresponses

from hummingbot.connector.derivative.bitget_perpetual import (
    bitget_perpetual_constants as CONSTANTS,
    bitget_perpetual_web_utils as web_utils,
)


class BitgetPerpetualWebUtilsTest(unittest.IsolatedAsyncioTestCase):

    async def async_run_with_timeout(self, coroutine: Awaitable, timeout: float = 1):
        ret = await asyncio.wait_for(coroutine, timeout)
        return ret

    def test_get_rest_url_for_endpoint(self):
        endpoint = "/testEndpoint"
        url = web_utils.get_rest_url_for_endpoint(endpoint)
        self.assertEqual("https://api.bitget.com/testEndpoint", url)

    @aioresponses()
    async def test_get_current_server_time(self, api_mock):
        url = web_utils.public_rest_url(path_url=CONSTANTS.SERVER_TIME_PATH_URL)
        data = {
            "flag": True,
            "requestTime": 1640001112223}

        api_mock.get(url=url, status=400, body=json.dumps(data))

        time = await self.async_run_with_timeout(web_utils.get_current_server_time())

        self.assertEqual(data["requestTime"], time)
