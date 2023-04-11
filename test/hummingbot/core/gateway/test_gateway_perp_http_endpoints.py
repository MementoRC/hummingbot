import asyncio
import unittest
from contextlib import ExitStack
from decimal import Decimal
from os.path import join, realpath
from test.mock.http_recorder import HttpPlayer
from test.utilities_for_async_tests import async_to_sync
from typing import Any, Dict
from unittest.mock import patch

from aiohttp import ClientSession

from hummingbot.core.data_type.common import PositionSide
from hummingbot.core.gateway.gateway_http_client import GatewayHttpClient


class GatewayHttpClientUnitTest(unittest.TestCase):
    _db_path: str
    _http_player: HttpPlayer
    _patch_stack: ExitStack
    _main_loop: asyncio.AbstractEventLoop
    _ev_loop: asyncio.AbstractEventLoop

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._main_loop = asyncio.get_event_loop()
        cls._ev_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls._ev_loop)
        cls._db_path = realpath(join(__file__, "../fixtures/gateway_perp_http_client_fixture.db"))
        cls._http_player = HttpPlayer(cls._db_path)
        cls._patch_stack = ExitStack()
        cls._patch_stack.enter_context(cls._http_player.patch_aiohttp_client())
        cls._patch_stack.enter_context(
            patch("hummingbot.core.gateway.gateway_http_client.GatewayHttpClient._http_client", return_value=ClientSession())
        )
        GatewayHttpClient.get_instance().base_url = "https://localhost:5000"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._patch_stack.close()
        cls._ev_loop.stop()
        cls._ev_loop.close()
        asyncio.set_event_loop(cls._main_loop)
        super().tearDownClass()

    @async_to_sync
    async def test_gateway_list(self):
        result: Dict[str, Any] = await GatewayHttpClient.get_instance().get_perp_markets(
            "ethereum",
            "optimism",
            "perp",
        )
        self.assertTrue(isinstance(result["pairs"], list))

    @async_to_sync
    async def test_gateway_status(self):
        result: Dict[str, Any] = await GatewayHttpClient.get_instance().get_perp_market_status(
            "ethereum",
            "optimism",
            "perp",
            "AAVE",
            "USD",
        )
        self.assertTrue(isinstance(result, dict))
        self.assertTrue(result["isActive"])

    @async_to_sync
    async def test_gateway_price(self):
        result: Dict[str, Any] = await GatewayHttpClient.get_instance().get_perp_market_price(
            "ethereum",
            "optimism",
            "perp",
            "AAVE",
            "USD",
            Decimal("0.1"),
            PositionSide.LONG,
        )
        self.assertTrue(isinstance(result, dict))
        self.assertEqual("72.3961573502110110555952936205735574981841", result["markPrice"])
        self.assertEqual("72.46", result["indexPrice"])
        self.assertEqual("72.71790773", result["indexTwapPrice"])

    @async_to_sync
    async def test_perp_balance(self):
        result: Dict[str, Any] = await GatewayHttpClient.get_instance().amm_perp_balance(
            "ethereum",
            "optimism",
            "perp",
            "0xefB7Be8631d154d4C0ad8676FEC0897B2894FE8F",
        )
        self.assertEqual("209.992983", result["balance"])

    @async_to_sync
    async def test_perp_open(self):
        result: Dict[str, Any] = await GatewayHttpClient.get_instance().amm_perp_open(
            "ethereum",
            "optimism",
            "perp",
            "0xefB7Be8631d154d4C0ad8676FEC0897B2894FE8F",
            "AAVE",
            "USD",
            PositionSide.LONG,
            Decimal("0.1"),
            Decimal("63"),
        )
        self.assertEqual("0.100000000000000000", result["amount"])
        self.assertEqual("0x48e65c8888282f268154146aa810a0da9bf10a3ffc2f98a298cf48a0fee864ac",      # noqa: mock
                         result["txHash"])

    @async_to_sync
    async def test_gateway_perp_position(self):
        result: Dict[str, Any] = await GatewayHttpClient.get_instance().get_perp_position(
            "ethereum",
            "optimism",
            "perp",
            "0xefB7Be8631d154d4C0ad8676FEC0897B2894FE8F",
            "AAVE",
            "USD",
        )
        self.assertTrue(isinstance(result, dict))
        self.assertEqual("LONG", result["positionSide"])
        self.assertEqual("AAVEUSD", result["tickerSymbol"])
        self.assertEqual("72.54886162448166764354", result["entryPrice"])

    @async_to_sync
    async def test_perp_close(self):
        result: Dict[str, Any] = await GatewayHttpClient.get_instance().amm_perp_close(
            "ethereum",
            "optimism",
            "perp",
            "0xefB7Be8631d154d4C0ad8676FEC0897B2894FE8F",
            "AAVE",
            "USD",
        )
        self.assertEqual("0x4ad832e07778d077af763e96395b4924ae84634a30b296f6f450d7c54e599da5",      # noqa: mock
                         result["txHash"])
