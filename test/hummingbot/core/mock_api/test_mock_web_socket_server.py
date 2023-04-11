import asyncio
import json
import unittest
from test.utilities_for_async_tests import async_to_sync
from unittest.mock import MagicMock, patch

import aiohttp

from hummingbot.core.mock_api.mock_web_socket_server import MockWebSocketServerFactory


class MockWebSocketServerFactoryTest(unittest.TestCase):
    ws_server = None
    _mock = None
    _patcher = None
    _main_loop: asyncio.AbstractEventLoop
    _ev_loop: asyncio.AbstractEventLoop

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.ws_server = MockWebSocketServerFactory.start_new_server("wss://www.google.com/ws/")

        cls._main_loop = asyncio.get_event_loop()
        cls._ev_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls._ev_loop)

    def tearDown(self) -> None:
        if self._patcher is not None:
            self._patcher.stop()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.ws_server.stop()
        cls._ev_loop.stop()
        cls._ev_loop.close()
        asyncio.set_event_loop(cls._main_loop)
        super().tearDownClass()

    async def asyncSetUp(self) -> None:
        self.session = aiohttp.ClientSession()
        await MockWebSocketServerFactory.set_original_ws_connect(self.session.ws_connect)

        self._patcher = patch("aiohttp.client.ClientSession.ws_connect", autospec=True)
        self._mock = self._patcher.start()
        self._mock.side_effect = MockWebSocketServerFactory.reroute_ws_connect

        # need to wait a bit for the server to be available
        await asyncio.wait_for(MockWebSocketServerFactoryTest.ws_server.wait_til_started(), 1)

    @async_to_sync
    async def test_web_socket(self):
        uri = "wss://www.google.com/ws/"

        await self.asyncSetUp()

        # Retry up to 3 times if there is any error connecting to the mock server address and port
        for retry_attempt in range(3):
            try:
                async with self.session.ws_connect(uri) as websocket:
                    await MockWebSocketServerFactory.send_str(uri, "aaa")
                    answer = await websocket.receive_str()
                    self.assertEqual("aaa", answer)

                    await MockWebSocketServerFactory.send_json(uri, data={"foo": "bar"})
                    answer = await websocket.receive_str()
                    answer = json.loads(answer)
                    self.assertEqual(answer["foo"], "bar")

                    await self.ws_server.websocket.send_str("xxx")
                    answer = await websocket.receive_str()
                    self.assertEqual("xxx", answer)
            except OSError:
                if retry_attempt == 2:
                    raise
                # Continue retrying
                continue

            # Stop the retries cycle
            break

        await self.session.close()

    @async_to_sync
    async def _test_start_new_server_with_invalid_uri(self):
        await self.asyncSetUp()
        server = MockWebSocketServerFactory.start_new_server("invalid_uri")

        with self.assertRaises(aiohttp.client_exceptions.ClientConnectorError):
            async with self.session.ws_connect("invalid_uri"):
                pass

        server.stop()
        await self.session.close()

    @async_to_sync
    async def test_reroute_ws_connect_invalid_uri(self):
        await self.asyncSetUp()
        with self.assertRaises(ValueError):
            await MockWebSocketServerFactory.reroute_ws_connect(self.session.ws_connect, "invalid_uri")
        await self.session.close()

    @async_to_sync
    async def _test_send_str_invalid_uri(self):
        await self.asyncSetUp()
        with self.assertRaises(KeyError):
            await MockWebSocketServerFactory.send_str("invalid_uri", "test")
        await self.session.close()

    @async_to_sync
    async def _test_send_json_invalid_uri(self):
        await self.asyncSetUp()
        with self.assertRaises(KeyError):
            await MockWebSocketServerFactory.send_json("invalid_uri", data={"test": "data"})
        await self.session.close()

    @async_to_sync
    async def _test_web_socket_close(self):
        uri = "wss://www.google.com/ws/"

        await self.asyncSetUp()

        # Retry up to 3 times if there is any error connecting to the mock server address and port
        for retry_attempt in range(3):
            try:
                async with self.session.ws_connect(uri) as websocket:
                    await MockWebSocketServerFactory.send_str(uri, "closing")
                    answer = await websocket.receive_str()
                    self.assertEqual("closing", answer)

                    await websocket.close()

                    msg = await websocket.receive()
                    self.assertEqual(msg.type, aiohttp.WSMsgType.CLOSED)
                    self.assertIsNone(msg.data)

            except OSError:
                if retry_attempt == 2:
                    raise
                # Continue retrying
                continue

            # Stop the retries cycle
            break

        await self.session.close()

    @async_to_sync
    async def _test_web_socket_exception_handling(self):
        uri = "wss://www.google.com/ws/"

        await self.asyncSetUp()

        self.ws_server.websocket.send_str = MagicMock(side_effect=OSError("Error sending data"))

        # Retry up to 3 times if there is any error connecting to the mock server address and port
        for retry_attempt in range(3):
            try:
                async with self.session.ws_connect(uri):
                    with self.assertRaises(OSError):
                        await self.ws_server.websocket.send_str("error")
            except OSError:
                if retry_attempt == 2:
                    raise
                # Continue retrying
                continue

            # Stop the retries cycle
            break

        await self.session.close()


if __name__ == '__main__':
    unittest.main()
