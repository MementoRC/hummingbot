import asyncio
import json
import unittest.mock

import requests
from aiohttp import ClientSession

from hummingbot.core.mock_api.mock_web_server import MockWebServer


class MockWebServerTest(unittest.IsolatedAsyncioTestCase):
    _req_url_mock = None
    _req_patcher = None
    _url_mock = None
    web_app: MockWebServer
    host: str
    _patcher: unittest.mock.patch

    @classmethod
    def setUpClass(cls) -> None:
        cls.web_app: MockWebServer = MockWebServer.get_instance()
        cls.host = "www.google.com"
        cls.web_app.add_host_to_mock(cls.host)
        cls.web_app.start()
        cls._patcher = unittest.mock.patch("aiohttp.client.URL")
        cls._url_mock = cls._patcher.start()
        cls._url_mock.side_effect = MockWebServer.reroute_local

        cls._req_patcher = unittest.mock.patch.object(requests.Session, "request", autospec=True)
        cls._req_url_mock = cls._req_patcher.start()
        cls._req_url_mock.side_effect = MockWebServer.reroute_request

    async def asyncSetUp(self) -> None:
        # need to wait a bit for the server to be available
        await asyncio.wait_for(MockWebServerTest.web_app.wait_til_started(), 1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.web_app.stop()
        cls._patcher.stop()
        cls._req_patcher.stop()

    async def _test_web_app_response(self):
        self.web_app.clear_responses()
        self.web_app.update_response("get", self.host, "/", data=self.web_app.TEST_RESPONSE, is_json=False)
        async with ClientSession() as client:
            async with client.get("http://www.google.com/") as resp:
                text: str = await resp.text()
                print(text)
                self.assertEqual(self.web_app.TEST_RESPONSE, text)

    async def test_web_app_response(self):
        await asyncio.wait_for(self._test_web_app_response(), 20)

    def test_get_request_response(self):
        self.web_app.clear_responses()
        self.web_app.update_response("get", self.host, "/", data=self.web_app.TEST_RESPONSE, is_json=False)
        r = requests.get("http://www.google.com/")
        self.assertEqual(self.web_app.TEST_RESPONSE, r.text)

    def test_update_response(self):
        self.web_app.clear_responses()
        self.web_app.update_response('get', 'www.google.com', '/', {"a": 1, "b": 2})
        r = requests.get("http://www.google.com/")
        r_json = json.loads(r.text)
        self.assertEqual(r_json["a"], 1)

        self.web_app.update_response('post', 'www.google.com', '/', "default")
        self.web_app.update_response('post', 'www.google.com', '/', {"a": 1, "b": 2}, params={"para_a": '11'})
        r = requests.post("http://www.google.com/", data={"para_a": 11, "para_b": 22})
        r_json = json.loads(r.text)
        self.assertEqual(r_json["a"], 1)

    def test_query_string(self):
        self.web_app.clear_responses()
        self.web_app.update_response('get', 'www.google.com', '/', "default")
        self.web_app.update_response('get', 'www.google.com', '/', {"a": 1}, params={"qs1": "1"})
        r = requests.get("http://www.google.com/?qs1=1")
        r_json = json.loads(r.text)
        self.assertEqual(r_json["a"], 1)


if __name__ == '__main__':
    unittest.main()
