"""
Unit tests for ExecutorFilter dataclass.
Tests cover construction, field defaults, equality, and basic filtering behavior.
"""

import unittest
from decimal import Decimal

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.controllers.controller_base import ExecutorFilter
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType


class TestExecutorFilter(unittest.TestCase):
    """Test cases for ExecutorFilter dataclass."""

    def test_default_construction(self):
        """Test ExecutorFilter can be constructed with all defaults (None)."""
        filter_obj = ExecutorFilter()
        self.assertIsNone(filter_obj.executor_ids)
        self.assertIsNone(filter_obj.connector_names)
        self.assertIsNone(filter_obj.trading_pairs)
        self.assertIsNone(filter_obj.executor_types)
        self.assertIsNone(filter_obj.statuses)
        self.assertIsNone(filter_obj.sides)
        self.assertIsNone(filter_obj.is_active)
        self.assertIsNone(filter_obj.is_trading)
        self.assertIsNone(filter_obj.close_types)
        self.assertIsNone(filter_obj.controller_ids)
        self.assertIsNone(filter_obj.min_pnl_pct)
        self.assertIsNone(filter_obj.max_pnl_pct)
        self.assertIsNone(filter_obj.min_pnl_quote)
        self.assertIsNone(filter_obj.max_pnl_quote)
        self.assertIsNone(filter_obj.min_timestamp)
        self.assertIsNone(filter_obj.max_timestamp)
        self.assertIsNone(filter_obj.min_close_timestamp)
        self.assertIsNone(filter_obj.max_close_timestamp)

    def test_construction_with_executor_ids(self):
        """Test ExecutorFilter construction with executor_ids."""
        executor_ids = ["exec1", "exec2", "exec3"]
        filter_obj = ExecutorFilter(executor_ids=executor_ids)
        self.assertEqual(filter_obj.executor_ids, executor_ids)
        self.assertIsNone(filter_obj.connector_names)

    def test_construction_with_connector_names(self):
        """Test ExecutorFilter construction with connector_names."""
        connector_names = ["binance", "coinbase"]
        filter_obj = ExecutorFilter(connector_names=connector_names)
        self.assertEqual(filter_obj.connector_names, connector_names)
        self.assertIsNone(filter_obj.executor_ids)

    def test_construction_with_trading_pairs(self):
        """Test ExecutorFilter construction with trading_pairs."""
        trading_pairs = ["BTC-USDT", "ETH-USDT"]
        filter_obj = ExecutorFilter(trading_pairs=trading_pairs)
        self.assertEqual(filter_obj.trading_pairs, trading_pairs)
        self.assertIsNone(filter_obj.connector_names)

    def test_construction_with_executor_types(self):
        """Test ExecutorFilter construction with executor_types."""
        executor_types = ["PositionExecutor", "DCAExecutor"]
        filter_obj = ExecutorFilter(executor_types=executor_types)
        self.assertEqual(filter_obj.executor_types, executor_types)

    def test_construction_with_statuses(self):
        """Test ExecutorFilter construction with statuses."""
        statuses = [RunnableStatus.RUNNING, RunnableStatus.TERMINATED]
        filter_obj = ExecutorFilter(statuses=statuses)
        self.assertEqual(filter_obj.statuses, statuses)

    def test_construction_with_sides(self):
        """Test ExecutorFilter construction with sides (TradeType)."""
        sides = [TradeType.BUY, TradeType.SELL]
        filter_obj = ExecutorFilter(sides=sides)
        self.assertEqual(filter_obj.sides, sides)

    def test_construction_with_boolean_fields(self):
        """Test ExecutorFilter construction with boolean fields."""
        filter_obj = ExecutorFilter(is_active=True, is_trading=False)
        self.assertTrue(filter_obj.is_active)
        self.assertFalse(filter_obj.is_trading)

    def test_construction_with_close_types(self):
        """Test ExecutorFilter construction with close_types."""
        close_types = [CloseType.COMPLETED, CloseType.EARLY_STOP]
        filter_obj = ExecutorFilter(close_types=close_types)
        self.assertEqual(filter_obj.close_types, close_types)

    def test_construction_with_controller_ids(self):
        """Test ExecutorFilter construction with controller_ids."""
        controller_ids = ["controller1", "controller2"]
        filter_obj = ExecutorFilter(controller_ids=controller_ids)
        self.assertEqual(filter_obj.controller_ids, controller_ids)

    def test_construction_with_pnl_pct_bounds(self):
        """Test ExecutorFilter construction with PNL percentage bounds."""
        min_pnl = Decimal("-0.05")
        max_pnl = Decimal("0.10")
        filter_obj = ExecutorFilter(min_pnl_pct=min_pnl, max_pnl_pct=max_pnl)
        self.assertEqual(filter_obj.min_pnl_pct, min_pnl)
        self.assertEqual(filter_obj.max_pnl_pct, max_pnl)

    def test_construction_with_pnl_quote_bounds(self):
        """Test ExecutorFilter construction with PNL quote bounds."""
        min_pnl = Decimal("-100.0")
        max_pnl = Decimal("500.0")
        filter_obj = ExecutorFilter(min_pnl_quote=min_pnl, max_pnl_quote=max_pnl)
        self.assertEqual(filter_obj.min_pnl_quote, min_pnl)
        self.assertEqual(filter_obj.max_pnl_quote, max_pnl)

    def test_construction_with_timestamp_bounds(self):
        """Test ExecutorFilter construction with timestamp bounds."""
        min_ts = 1640995200.0
        max_ts = 1672531200.0
        filter_obj = ExecutorFilter(min_timestamp=min_ts, max_timestamp=max_ts)
        self.assertEqual(filter_obj.min_timestamp, min_ts)
        self.assertEqual(filter_obj.max_timestamp, max_ts)

    def test_construction_with_close_timestamp_bounds(self):
        """Test ExecutorFilter construction with close timestamp bounds."""
        min_close_ts = 1640995200.0
        max_close_ts = 1672531200.0
        filter_obj = ExecutorFilter(min_close_timestamp=min_close_ts, max_close_timestamp=max_close_ts)
        self.assertEqual(filter_obj.min_close_timestamp, min_close_ts)
        self.assertEqual(filter_obj.max_close_timestamp, max_close_ts)

    def test_construction_with_all_fields(self):
        """Test ExecutorFilter construction with all fields specified."""
        filter_obj = ExecutorFilter(
            executor_ids=["exec1"],
            connector_names=["binance"],
            trading_pairs=["BTC-USDT"],
            executor_types=["PositionExecutor"],
            statuses=[RunnableStatus.RUNNING],
            sides=[TradeType.BUY],
            is_active=True,
            is_trading=True,
            close_types=[CloseType.COMPLETED],
            controller_ids=["ctrl1"],
            min_pnl_pct=Decimal("0.0"),
            max_pnl_pct=Decimal("1.0"),
            min_pnl_quote=Decimal("0.0"),
            max_pnl_quote=Decimal("1000.0"),
            min_timestamp=1640995200.0,
            max_timestamp=1672531200.0,
            min_close_timestamp=1640995200.0,
            max_close_timestamp=1672531200.0,
        )

        self.assertEqual(filter_obj.executor_ids, ["exec1"])
        self.assertEqual(filter_obj.connector_names, ["binance"])
        self.assertEqual(filter_obj.trading_pairs, ["BTC-USDT"])
        self.assertEqual(filter_obj.executor_types, ["PositionExecutor"])
        self.assertEqual(filter_obj.statuses, [RunnableStatus.RUNNING])
        self.assertEqual(filter_obj.sides, [TradeType.BUY])
        self.assertTrue(filter_obj.is_active)
        self.assertTrue(filter_obj.is_trading)
        self.assertEqual(filter_obj.close_types, [CloseType.COMPLETED])
        self.assertEqual(filter_obj.controller_ids, ["ctrl1"])
        self.assertEqual(filter_obj.min_pnl_pct, Decimal("0.0"))
        self.assertEqual(filter_obj.max_pnl_pct, Decimal("1.0"))
        self.assertEqual(filter_obj.min_pnl_quote, Decimal("0.0"))
        self.assertEqual(filter_obj.max_pnl_quote, Decimal("1000.0"))
        self.assertEqual(filter_obj.min_timestamp, 1640995200.0)
        self.assertEqual(filter_obj.max_timestamp, 1672531200.0)
        self.assertEqual(filter_obj.min_close_timestamp, 1640995200.0)
        self.assertEqual(filter_obj.max_close_timestamp, 1672531200.0)

    def test_equality_same_filters(self):
        """Test equality of two ExecutorFilter objects with same values."""
        filter1 = ExecutorFilter(executor_ids=["exec1"], connector_names=["binance"])
        filter2 = ExecutorFilter(executor_ids=["exec1"], connector_names=["binance"])
        self.assertEqual(filter1, filter2)

    def test_equality_different_filters(self):
        """Test inequality of two ExecutorFilter objects with different values."""
        filter1 = ExecutorFilter(executor_ids=["exec1"])
        filter2 = ExecutorFilter(executor_ids=["exec2"])
        self.assertNotEqual(filter1, filter2)

    def test_equality_both_default(self):
        """Test equality of two default ExecutorFilter objects."""
        filter1 = ExecutorFilter()
        filter2 = ExecutorFilter()
        self.assertEqual(filter1, filter2)

    def test_empty_list_vs_none(self):
        """Test that empty list and None are different."""
        filter_none = ExecutorFilter(executor_ids=None)
        filter_empty = ExecutorFilter(executor_ids=[])
        self.assertNotEqual(filter_none, filter_empty)
        self.assertIsNone(filter_none.executor_ids)
        self.assertEqual(filter_empty.executor_ids, [])

    def test_dataclass_repr(self):
        """Test that ExecutorFilter has a meaningful repr."""
        filter_obj = ExecutorFilter(executor_ids=["exec1"])
        repr_str = repr(filter_obj)
        self.assertIn("ExecutorFilter", repr_str)
        self.assertIn("exec1", repr_str)

    def test_multiple_list_fields_with_values(self):
        """Test ExecutorFilter with multiple list fields populated."""
        filter_obj = ExecutorFilter(
            executor_ids=["exec1", "exec2"],
            connector_names=["binance", "coinbase"],
            trading_pairs=["BTC-USDT", "ETH-USDT"],
        )
        self.assertEqual(len(filter_obj.executor_ids), 2)
        self.assertEqual(len(filter_obj.connector_names), 2)
        self.assertEqual(len(filter_obj.trading_pairs), 2)

    def test_pnl_decimal_precision(self):
        """Test that Decimal precision is maintained in PNL fields."""
        precise_pnl = Decimal("0.123456789")
        filter_obj = ExecutorFilter(min_pnl_pct=precise_pnl)
        self.assertEqual(filter_obj.min_pnl_pct, precise_pnl)
        self.assertEqual(str(filter_obj.min_pnl_pct), "0.123456789")

    def test_negative_pnl_values(self):
        """Test ExecutorFilter with negative PNL values."""
        filter_obj = ExecutorFilter(
            min_pnl_pct=Decimal("-0.50"),
            min_pnl_quote=Decimal("-1000.0"),
        )
        self.assertEqual(filter_obj.min_pnl_pct, Decimal("-0.50"))
        self.assertEqual(filter_obj.min_pnl_quote, Decimal("-1000.0"))
