import asyncio
from decimal import Decimal
import unittest

import pytest

pytest.importorskip("market_connector")

from data_type_primitives import OrderBookEntry, OrderBookSnapshot  # noqa: E402

from hummingbot.connector.market_connector.market_connector_exchange import (  # noqa: E402
    PRE_START_STATUS_KEYS,
    MarketConnectorExchange,
)
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType  # noqa: E402
from hummingbot.core.network_iterator import NetworkStatus  # noqa: E402

from .fakes import PAIR, FakeDescriptor  # noqa: E402


def _snapshot() -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=1000,
        bids=[OrderBookEntry(price=Decimal("99"), quantity=Decimal("2"))],
        asks=[OrderBookEntry(price=Decimal("101"), quantity=Decimal("3"))],
    )


class MarketConnectorExchangeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.descriptor = FakeDescriptor()
        self.exchange = MarketConnectorExchange(
            descriptor=self.descriptor,
            config={"k": "v"},
            connector_name="fake_mc",
            trading_pairs=[PAIR],
            core_options={"unknown_order_grace": 60.0},
            order_book_refresh_interval=0.01,
        )

    async def test_constructor_does_no_io_and_does_not_call_create(self):
        self.assertEqual([], self.descriptor.create_calls)
        self.assertEqual("fake_mc", self.exchange.name)
        self.assertEqual([PAIR], self.exchange.trading_pairs)

    async def test_pre_start_safe_defaults(self):
        ex = self.exchange
        self.assertFalse(ex.ready)
        self.assertEqual({key: False for key in PRE_START_STATUS_KEYS}, ex.status_dict)
        self.assertEqual(Decimal(0), ex.get_balance("ETH"))
        self.assertEqual(Decimal(0), ex.get_available_balance("ETH"))
        self.assertTrue(ex.get_price_by_type(PAIR, PriceType.MidPrice).is_nan())
        self.assertTrue(ex.get_price(PAIR, True).is_nan())
        self.assertEqual({}, ex.trading_rules)
        self.assertEqual({}, ex.in_flight_orders)
        self.assertEqual([], ex.limit_orders)
        self.assertEqual(NetworkStatus.NOT_CONNECTED, await ex.check_network())

    async def test_start_network_creates_starts_and_tracks(self):
        await self.exchange.start_network()
        config, bus, options = self.descriptor.create_calls[0]
        self.assertEqual({"k": "v"}, config)
        self.assertIsNotNone(bus)
        self.assertEqual({"unknown_order_grace": 60.0}, options)
        self.assertEqual(1, self.descriptor.handle.started)
        self.assertEqual([PAIR], self.descriptor.core.tracked)
        self.assertEqual(NetworkStatus.CONNECTED, await self.exchange.check_network())
        await self.exchange.stop_network()

    async def test_refresh_loop_updates_books_and_survives_errors(self):
        self.descriptor.core.fail_refresh = True
        await self.exchange.start_network()
        await asyncio.sleep(0.05)
        await self.exchange.stop_network()
        self.assertGreaterEqual(len(self.descriptor.core.updated), 2)

    async def test_stop_network_closes_once_and_clears(self):
        await self.exchange.start_network()
        await self.exchange.stop_network()
        await self.exchange.stop_network()
        self.assertEqual(1, self.descriptor.handle.closed)
        self.assertEqual(NetworkStatus.NOT_CONNECTED, await self.exchange.check_network())
        self.assertFalse(self.exchange.ready)

    async def test_status_dict_passthrough_and_ready(self):
        await self.exchange.start_network()
        self.assertEqual(5, len(self.exchange.status_dict))
        self.assertIn("user_stream_initialized", self.exchange.status_dict)
        self.assertTrue(self.exchange.ready)
        self.descriptor.core.status["user_stream_initialized"] = False
        self.assertFalse(self.exchange.status_dict["user_stream_initialized"])
        self.assertFalse(self.exchange.ready)
        await self.exchange.stop_network()

    async def test_order_book_and_prices(self):
        await self.exchange.start_network()
        core = self.descriptor.core
        core.snapshot = _snapshot()
        book = self.exchange.get_order_book(PAIR)
        self.assertEqual(99.0, book.get_price(False))
        self.assertEqual(Decimal("101"), self.exchange.get_price(PAIR, True))
        self.assertEqual(Decimal("99"), self.exchange.get_price(PAIR, False))
        self.assertEqual([PAIR], list(self.exchange.order_books))
        # core has no cached prices yet: falls back to the book
        self.assertEqual(Decimal("100"), self.exchange.get_price_by_type(PAIR, PriceType.MidPrice))
        core.prices[PriceType.BestBid] = Decimal("98")
        self.assertEqual(Decimal("98"), self.exchange.get_price_by_type(PAIR, PriceType.BestBid))
        await self.exchange.stop_network()

    async def test_balances(self):
        await self.exchange.start_network()
        self.assertEqual(Decimal("2"), self.exchange.get_balance("ETH"))
        self.assertEqual(Decimal("1.5"), self.exchange.get_available_balance("ETH"))
        self.assertEqual({"ETH": Decimal("2"), "USDC": Decimal(0)}, self.exchange.get_all_balances())
        await self.exchange.stop_network()

    async def test_quantization_and_quantums(self):
        await self.exchange.start_network()
        self.assertEqual(Decimal("1.235"), self.exchange.quantize_order_amount(PAIR, Decimal("1.2349")))
        self.assertEqual(Decimal("10.12"), self.exchange.quantize_order_price(PAIR, Decimal("10.1234")))
        self.assertEqual(Decimal("0.01"), self.exchange.get_order_price_quantum(PAIR, Decimal(1)))
        self.assertEqual(Decimal("0.001"), self.exchange.get_order_size_quantum(PAIR, Decimal(1)))
        await self.exchange.stop_network()

    async def test_trading_rules_are_converted(self):
        await self.exchange.start_network()
        rules = self.exchange.trading_rules
        self.assertEqual(Decimal("0.01"), rules[PAIR].min_order_size)
        self.assertEqual(Decimal("0.001"), rules[PAIR].min_base_amount_increment)
        self.assertIs(rules[PAIR], self.exchange.trading_rules[PAIR])
        await self.exchange.stop_network()

    async def test_get_fee_uses_descriptor_fees(self):
        maker = self.exchange.get_fee("ETH", "USDC", OrderType.LIMIT_MAKER, TradeType.BUY, Decimal(1))
        taker = self.exchange.get_fee("ETH", "USDC", OrderType.MARKET, TradeType.SELL, Decimal(1))
        self.assertEqual(Decimal("0.0025"), maker.percent)
        self.assertEqual(Decimal("0.004"), taker.percent)

    async def test_supported_order_types_from_capabilities(self):
        self.assertEqual(
            {
                OrderType.MARKET,
                OrderType.LIMIT,
                OrderType.STOP_LOSS,
                OrderType.TAKE_PROFIT,
                OrderType.STOP_LOSS_LIMIT,
                OrderType.TAKE_PROFIT_LIMIT,
            },
            set(self.exchange.supported_order_types()),
        )
