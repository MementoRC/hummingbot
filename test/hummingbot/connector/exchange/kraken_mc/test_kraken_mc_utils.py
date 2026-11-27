from decimal import Decimal
import unittest

import pytest

from hummingbot.client.settings import AllConnectorSettings
from hummingbot.connector.exchange.kraken_mc import kraken_mc_utils as utils


class KrakenMcUtilsTests(unittest.TestCase):
    def test_default_fees(self):
        self.assertEqual(Decimal("0.0025"), utils.DEFAULT_FEES.maker_percent_fee_decimal)
        self.assertEqual(Decimal("0.004"), utils.DEFAULT_FEES.taker_percent_fee_decimal)

    def test_keys_fields(self):
        fields = set(type(utils.KEYS).model_fields)
        self.assertTrue({"kraken_mc_api_key", "kraken_mc_secret_key", "kraken_mc_api_tier"} <= fields)
        self.assertEqual("kraken_mc", utils.KEYS.connector)

    def test_tier_validator(self):
        cfg = utils.KrakenMcConfigMap
        ok = cfg(kraken_mc_api_key="a", kraken_mc_secret_key="b", kraken_mc_api_tier="Pro")
        self.assertEqual("Pro", ok.kraken_mc_api_tier)
        with self.assertRaises(ValueError):
            cfg(kraken_mc_api_key="a", kraken_mc_secret_key="b", kraken_mc_api_tier="Gold")

    def test_host_slot_default_and_bounds(self):
        cfg = utils.KrakenMcConfigMap
        base = {"kraken_mc_api_key": "a", "kraken_mc_secret_key": "b"}
        self.assertEqual(0, cfg(**base).kraken_mc_host_slot)
        self.assertEqual(99, cfg(**base, kraken_mc_host_slot=99).kraken_mc_host_slot)
        for bad in (-1, 100):
            with self.assertRaises(ValueError):
                cfg(**base, kraken_mc_host_slot=bad)

    def test_example_pair_matches_descriptor(self):
        registry = pytest.importorskip("market_connector.hb_compat.registry")
        self.assertEqual(registry.get_exchange("kraken").example_pair, utils.EXAMPLE_PAIR)

    def test_connector_is_discovered(self):
        settings = AllConnectorSettings.create_connector_settings()
        setting = settings["kraken_mc"]
        self.assertEqual("hummingbot.connector.exchange.kraken_mc.kraken_mc_exchange", setting.module_path())
        self.assertEqual("KrakenMcExchange", setting.class_name())
        self.assertTrue(setting.centralised)
