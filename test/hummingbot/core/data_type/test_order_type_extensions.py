from unittest import TestCase

from hummingbot.core.data_type.common import OrderType


class OrderTypeExtensionsTest(TestCase):
    # --- Existence tests ---

    def test_stop_loss_exists(self):
        self.assertIn("STOP_LOSS", OrderType.__members__)

    def test_take_profit_exists(self):
        self.assertIn("TAKE_PROFIT", OrderType.__members__)

    def test_trailing_stop_exists(self):
        self.assertIn("TRAILING_STOP", OrderType.__members__)

    # --- Original types preserved ---

    def test_original_types_preserved(self):
        self.assertEqual(OrderType.MARKET.value, 1)
        self.assertEqual(OrderType.LIMIT.value, 2)
        self.assertEqual(OrderType.LIMIT_MAKER.value, 3)

    # --- is_limit_type ---

    def test_is_limit_type_includes_stop_loss_limit(self):
        self.assertTrue(OrderType.STOP_LOSS_LIMIT.is_limit_type())

    def test_is_limit_type_includes_take_profit_limit(self):
        self.assertTrue(OrderType.TAKE_PROFIT_LIMIT.is_limit_type())

    def test_is_limit_type_includes_trailing_stop_limit(self):
        self.assertTrue(OrderType.TRAILING_STOP_LIMIT.is_limit_type())

    def test_is_limit_type_excludes_stop_loss(self):
        self.assertFalse(OrderType.STOP_LOSS.is_limit_type())

    def test_is_limit_type_excludes_take_profit(self):
        self.assertFalse(OrderType.TAKE_PROFIT.is_limit_type())

    def test_is_limit_type_excludes_trailing_stop(self):
        self.assertFalse(OrderType.TRAILING_STOP.is_limit_type())

    # --- is_delayed_market_type ---

    def test_is_delayed_market_type(self):
        self.assertTrue(OrderType.STOP_LOSS.is_delayed_market_type())
        self.assertTrue(OrderType.TAKE_PROFIT.is_delayed_market_type())
        self.assertTrue(OrderType.TRAILING_STOP.is_delayed_market_type())

    def test_is_delayed_market_type_false_market(self):
        self.assertFalse(OrderType.MARKET.is_delayed_market_type())

    def test_is_delayed_market_type_false_limit(self):
        self.assertFalse(OrderType.LIMIT.is_delayed_market_type())

    def test_is_delayed_market_type_false_limit_maker(self):
        self.assertFalse(OrderType.LIMIT_MAKER.is_delayed_market_type())

    def test_is_delayed_market_type_false_stop_loss_limit(self):
        self.assertFalse(OrderType.STOP_LOSS_LIMIT.is_delayed_market_type())

    # --- is_conditional_type ---

    def test_is_conditional_type(self):
        conditional_types = [
            OrderType.STOP_LOSS,
            OrderType.TAKE_PROFIT,
            OrderType.TRAILING_STOP,
            OrderType.STOP_LOSS_LIMIT,
            OrderType.TAKE_PROFIT_LIMIT,
            OrderType.TRAILING_STOP_LIMIT,
        ]
        for order_type in conditional_types:
            with self.subTest(order_type=order_type):
                self.assertTrue(order_type.is_conditional_type())

    def test_is_conditional_type_false_market(self):
        self.assertFalse(OrderType.MARKET.is_conditional_type())

    def test_is_conditional_type_false_limit(self):
        self.assertFalse(OrderType.LIMIT.is_conditional_type())

    def test_is_conditional_type_false_limit_maker(self):
        self.assertFalse(OrderType.LIMIT_MAKER.is_conditional_type())
