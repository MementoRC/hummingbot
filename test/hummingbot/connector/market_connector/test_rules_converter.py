from decimal import Decimal
import unittest

import pytest

pytest.importorskip("market_connector")

from data_type_primitives import TradingRules  # noqa: E402

from hummingbot.connector.market_connector.rules_converter import to_parent_trading_rule  # noqa: E402


class RulesConverterTests(unittest.TestCase):
    def test_full_rules_are_copied(self):
        rules = TradingRules(
            trading_pair="ETH-USDC",
            min_order_size=Decimal("0.01"),
            max_order_size=Decimal("100"),
            min_price_increment=Decimal("0.01"),
            min_base_amount_increment=Decimal("0.0001"),
            min_notional_size=Decimal("5"),
        )
        rule = to_parent_trading_rule(rules)
        self.assertEqual("ETH-USDC", rule.trading_pair)
        self.assertEqual(Decimal("0.01"), rule.min_order_size)
        self.assertEqual(Decimal("100"), rule.max_order_size)
        self.assertEqual(Decimal("0.01"), rule.min_price_increment)
        self.assertEqual(Decimal("0.0001"), rule.min_base_amount_increment)
        self.assertEqual(Decimal("5"), rule.min_notional_size)
        self.assertEqual("USDC", rule.buy_order_collateral_token)

    def test_unconstrained_fields_use_parent_defaults(self):
        rule = to_parent_trading_rule(TradingRules(trading_pair="ETH-USDC"))
        self.assertEqual(Decimal("1e56"), rule.max_order_size)
        self.assertGreater(rule.min_price_increment, 0)
        self.assertGreater(rule.min_base_amount_increment, 0)
