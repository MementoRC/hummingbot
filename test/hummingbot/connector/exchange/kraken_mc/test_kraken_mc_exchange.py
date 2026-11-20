from decimal import Decimal
import unittest

import pytest

pytest.importorskip("market_connector")

from hummingbot.connector.exchange.kraken_mc.kraken_mc_exchange import KrakenMcExchange  # noqa: E402


class KrakenMcExchangeTests(unittest.TestCase):
    def setUp(self):
        self.exchange = KrakenMcExchange(
            kraken_mc_api_key="key",
            kraken_mc_secret_key="secret",
            kraken_mc_api_tier="Pro",
            trading_pairs=["ETH-USDC"],
            client_config_map=object(),  # tolerated, ignored
        )

    def test_name_and_pairs(self):
        self.assertEqual("kraken_mc", self.exchange.name)
        self.assertEqual(["ETH-USDC"], self.exchange.trading_pairs)

    def test_keys_are_deprefixed_to_canonical(self):
        self.assertEqual(
            {"kraken_api_key": "key", "kraken_secret_key": "secret", "kraken_api_tier": "Pro"},
            self.exchange._config,
        )

    def test_core_options(self):
        self.assertEqual(60.0, self.exchange._core_options["unknown_order_grace"])
        self.assertTrue(self.exchange._core_options["own_trades_ws"])

    def test_descriptor_fees_and_not_ready_before_start(self):
        self.assertEqual(Decimal("0.0025"), self.exchange._descriptor.default_maker_fee)
        self.assertFalse(self.exchange.ready)
        self.assertEqual(5, len(self.exchange.status_dict))
