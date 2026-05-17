from unittest import TestCase

from hummingbot.core.data_type.common import GroupedSetDict, LazyDict, OrderType


class GroupedSetDictTests(TestCase):
    def setUp(self):
        self.dict = GroupedSetDict[str, str]()

    def test_add_or_update_new_key(self):
        self.dict.add_or_update("key1", "value1")
        self.assertEqual(self.dict["key1"], {"value1"})

    def test_add_or_update_existing_key(self):
        self.dict.add_or_update("key1", "value1")
        self.dict.add_or_update("key1", "value2")
        self.assertEqual(self.dict["key1"], {"value1", "value2"})

    def test_add_or_update_chaining(self):
        (
            self.dict.add_or_update("key1", "value1")
            .add_or_update("key1", "value2")
            .add_or_update("key1", "value2")  # This should be a no-op
            .add_or_update("key2", "value1")
        )
        self.assertEqual(self.dict["key1"], {"value1", "value2"})
        self.assertEqual(self.dict["key2"], {"value1"})

    def test_add_or_update_multiple_values(self):
        self.dict.add_or_update("key1", "value1", "value2", "value3")
        self.assertEqual(self.dict["key1"], {"value1", "value2", "value3"})

    def test_market_dict_type(self):
        market_dict = GroupedSetDict[str, set[str]]()
        market_dict.add_or_update("exchange1", "BTC-USDT")
        self.assertEqual(market_dict["exchange1"], {"BTC-USDT"})


class LambdaDictTests(TestCase):
    def setUp(self):
        self.dict = LazyDict[str, int]()

    def test_get_or_add_new_key(self):
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return 42

        value = self.dict.get_or_add("key1", factory)

        self.assertEqual(value, 42)
        self.assertEqual(call_count, 1)
        # Verify factory not called again on subsequent gets
        self.assertEqual(self.dict.get_or_add("key1", factory), 42)
        self.assertEqual(call_count, 1)

        # Verify factory is called again for new key
        self.assertEqual(self.dict.get_or_add("key2", factory), 42)
        self.assertEqual(call_count, 2)

    def test_get_or_add_existing_key(self):
        self.dict["key1"] = 42

        def factory():
            return 100

        value = self.dict.get_or_add("key1", factory)
        self.assertEqual(value, 42)
        self.assertEqual(self.dict["key1"], 42)

    def test_default_value_factory(self):
        call_count = 0

        def factory(key: str) -> int:
            nonlocal call_count
            call_count += 1
            return len(key)

        self.dict = LazyDict[str, int](default_value_factory=factory)
        self.assertEqual(self.dict["key1"], 4)
        self.assertEqual(call_count, 1)
        # Verify factory is not called again for existing key
        self.assertEqual(self.dict["key1"], 4)
        self.assertEqual(self.dict.get("key1"), 4)
        self.assertEqual(call_count, 1)
        # Verify factory is called again for new key
        self.assertEqual(self.dict["longer_key"], 10)
        self.assertEqual(self.dict.get("longer_key"), 10)
        self.assertEqual(call_count, 2)

    def test_missing_key_no_factory(self):
        with self.assertRaises(KeyError):
            _ = self.dict["nonexistent"]
        with self.assertRaises(KeyError):
            _ = self.dict.get("nonexistent")


class OrderTypeTests(TestCase):
    # is_limit_type — true cases
    def test_is_limit_type_for_limit(self):
        self.assertTrue(OrderType.LIMIT.is_limit_type())

    def test_is_limit_type_for_limit_maker(self):
        self.assertTrue(OrderType.LIMIT_MAKER.is_limit_type())

    def test_is_limit_type_for_stop_loss_limit(self):
        self.assertTrue(OrderType.STOP_LOSS_LIMIT.is_limit_type())

    def test_is_limit_type_for_take_profit_limit(self):
        self.assertTrue(OrderType.TAKE_PROFIT_LIMIT.is_limit_type())

    def test_is_limit_type_for_trailing_stop_limit(self):
        self.assertTrue(OrderType.TRAILING_STOP_LIMIT.is_limit_type())

    # is_limit_type — false cases (non-LIMIT and conditional-non-LIMIT)
    def test_is_limit_type_for_market(self):
        self.assertFalse(OrderType.MARKET.is_limit_type())

    def test_is_limit_type_for_amm_swap(self):
        self.assertFalse(OrderType.AMM_SWAP.is_limit_type())

    def test_is_limit_type_for_amm_add(self):
        self.assertFalse(OrderType.AMM_ADD.is_limit_type())

    def test_is_limit_type_for_amm_remove(self):
        self.assertFalse(OrderType.AMM_REMOVE.is_limit_type())

    def test_is_limit_type_for_stop_loss(self):
        self.assertFalse(OrderType.STOP_LOSS.is_limit_type())

    def test_is_limit_type_for_take_profit(self):
        self.assertFalse(OrderType.TAKE_PROFIT.is_limit_type())

    def test_is_limit_type_for_trailing_stop(self):
        self.assertFalse(OrderType.TRAILING_STOP.is_limit_type())

    # is_delayed_market_type — true cases (conditional, market-execution)
    def test_is_delayed_market_type_for_stop_loss(self):
        self.assertTrue(OrderType.STOP_LOSS.is_delayed_market_type())

    def test_is_delayed_market_type_for_take_profit(self):
        self.assertTrue(OrderType.TAKE_PROFIT.is_delayed_market_type())

    def test_is_delayed_market_type_for_trailing_stop(self):
        self.assertTrue(OrderType.TRAILING_STOP.is_delayed_market_type())

    # is_delayed_market_type — false cases (LIMIT variants and non-conditional)
    def test_is_delayed_market_type_for_market(self):
        self.assertFalse(OrderType.MARKET.is_delayed_market_type())

    def test_is_delayed_market_type_for_limit(self):
        self.assertFalse(OrderType.LIMIT.is_delayed_market_type())

    def test_is_delayed_market_type_for_stop_loss_limit(self):
        self.assertFalse(OrderType.STOP_LOSS_LIMIT.is_delayed_market_type())

    # is_conditional_type — true cases (all 6 new conditional members)
    def test_is_conditional_type_for_stop_loss(self):
        self.assertTrue(OrderType.STOP_LOSS.is_conditional_type())

    def test_is_conditional_type_for_take_profit(self):
        self.assertTrue(OrderType.TAKE_PROFIT.is_conditional_type())

    def test_is_conditional_type_for_trailing_stop(self):
        self.assertTrue(OrderType.TRAILING_STOP.is_conditional_type())

    def test_is_conditional_type_for_stop_loss_limit(self):
        self.assertTrue(OrderType.STOP_LOSS_LIMIT.is_conditional_type())

    def test_is_conditional_type_for_take_profit_limit(self):
        self.assertTrue(OrderType.TAKE_PROFIT_LIMIT.is_conditional_type())

    def test_is_conditional_type_for_trailing_stop_limit(self):
        self.assertTrue(OrderType.TRAILING_STOP_LIMIT.is_conditional_type())

    # is_conditional_type — false cases (non-conditional standard types)
    def test_is_conditional_type_for_market(self):
        self.assertFalse(OrderType.MARKET.is_conditional_type())

    def test_is_conditional_type_for_limit(self):
        self.assertFalse(OrderType.LIMIT.is_conditional_type())

    def test_is_conditional_type_for_limit_maker(self):
        self.assertFalse(OrderType.LIMIT_MAKER.is_conditional_type())

    def test_is_conditional_type_for_amm_swap(self):
        self.assertFalse(OrderType.AMM_SWAP.is_conditional_type())

    def test_order_type_enum_members(self):
        """Verify that all expected OrderType enum members exist (extended for conditional types)."""
        expected_members = {
            "MARKET",
            "LIMIT",
            "LIMIT_MAKER",
            "AMM_SWAP",
            "AMM_ADD",
            "AMM_REMOVE",
            "STOP_LOSS",
            "TAKE_PROFIT",
            "TRAILING_STOP",
            "STOP_LOSS_LIMIT",
            "TAKE_PROFIT_LIMIT",
            "TRAILING_STOP_LIMIT",
        }
        actual_members = {member.name for member in OrderType}
        self.assertEqual(actual_members, expected_members)

    def test_is_limit_type_true_count(self):
        """Verify exactly 5 OrderType members return True for is_limit_type() (LIMIT + LIMIT_MAKER + 3 conditional-LIMIT variants)."""
        limit_types = [ot for ot in OrderType if ot.is_limit_type()]
        self.assertEqual(len(limit_types), 5)
        self.assertEqual(
            set(limit_types),
            {
                OrderType.LIMIT,
                OrderType.LIMIT_MAKER,
                OrderType.STOP_LOSS_LIMIT,
                OrderType.TAKE_PROFIT_LIMIT,
                OrderType.TRAILING_STOP_LIMIT,
            },
        )

    def test_is_delayed_market_type_true_count(self):
        """Verify exactly 3 OrderType members return True for is_delayed_market_type()."""
        delayed_market_types = [ot for ot in OrderType if ot.is_delayed_market_type()]
        self.assertEqual(len(delayed_market_types), 3)
        self.assertEqual(
            set(delayed_market_types),
            {
                OrderType.STOP_LOSS,
                OrderType.TAKE_PROFIT,
                OrderType.TRAILING_STOP,
            },
        )

    def test_is_conditional_type_true_count(self):
        """Verify exactly 6 OrderType members return True for is_conditional_type()."""
        conditional_types = [ot for ot in OrderType if ot.is_conditional_type()]
        self.assertEqual(len(conditional_types), 6)
        self.assertEqual(
            set(conditional_types),
            {
                OrderType.STOP_LOSS,
                OrderType.TAKE_PROFIT,
                OrderType.TRAILING_STOP,
                OrderType.STOP_LOSS_LIMIT,
                OrderType.TAKE_PROFIT_LIMIT,
                OrderType.TRAILING_STOP_LIMIT,
            },
        )


if __name__ == "__main__":
    TestCase.main()
