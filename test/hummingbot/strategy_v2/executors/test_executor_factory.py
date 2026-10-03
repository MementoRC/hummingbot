"""Integration tests for ExecutorFactory decorator-based registration.

Verifies that all 8 executor types are registered, that create() dispatches
to the correct executor class, and that the orchestrator uses the factory.
"""

from decimal import Decimal
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from hummingbot.connector.exchange_py_base import ExchangePyBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.arbitrage_executor.arbitrage_executor import ArbitrageExecutor
from hummingbot.strategy_v2.executors.arbitrage_executor.data_types import ArbitrageExecutorConfig
from hummingbot.strategy_v2.executors.data_types import ConnectorPair, ExecutorConfigBase
from hummingbot.strategy_v2.executors.dca_executor.data_types import DCAExecutorConfig
from hummingbot.strategy_v2.executors.dca_executor.dca_executor import DCAExecutor
from hummingbot.strategy_v2.executors.executor_factory import ExecutorFactory
from hummingbot.strategy_v2.executors.grid_executor.data_types import GridExecutorConfig
from hummingbot.strategy_v2.executors.grid_executor.grid_executor import GridExecutor
from hummingbot.strategy_v2.executors.lp_executor.data_types import LPExecutorConfig
from hummingbot.strategy_v2.executors.lp_executor.lp_executor import LPExecutor
from hummingbot.strategy_v2.executors.order_executor.data_types import ExecutionStrategy, OrderExecutorConfig
from hummingbot.strategy_v2.executors.order_executor.order_executor import OrderExecutor
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.executors.position_executor.position_executor import PositionExecutor
from hummingbot.strategy_v2.executors.twap_executor.data_types import TWAPExecutorConfig
from hummingbot.strategy_v2.executors.twap_executor.twap_executor import TWAPExecutor
from hummingbot.strategy_v2.executors.xemm_executor.data_types import XEMMExecutorConfig
from hummingbot.strategy_v2.executors.xemm_executor.xemm_executor import XEMMExecutor

# All 8 registered (config_class, executor_class) pairs
EXPECTED_REGISTRY = {
    PositionExecutorConfig: PositionExecutor,
    DCAExecutorConfig: DCAExecutor,
    GridExecutorConfig: GridExecutor,
    TWAPExecutorConfig: TWAPExecutor,
    OrderExecutorConfig: OrderExecutor,
    XEMMExecutorConfig: XEMMExecutor,
    ArbitrageExecutorConfig: ArbitrageExecutor,
    LPExecutorConfig: LPExecutor,
}


def _make_mock_strategy(connectors=None):
    """Create a mock strategy with realistic connector setup."""
    strategy = MagicMock(spec=StrategyV2Base)
    connector = MagicMock(spec=ExchangePyBase)
    type(connector).trading_rules = PropertyMock(return_value={"ETH-USDT": TradingRule(trading_pair="ETH-USDT")})
    if connectors is None:
        connectors = {"binance": connector}
    strategy.connectors = connectors
    for conn in strategy.connectors.values():
        conn.supported_order_types.return_value = [OrderType.MARKET, OrderType.LIMIT, OrderType.LIMIT_MAKER]
    strategy.market_data_provider = MagicMock(spec=MarketDataProvider)
    strategy.market_data_provider.get_price_by_type = MagicMock(return_value=Decimal(230))
    strategy.controllers = {}
    strategy.markets = {"binance": {"ETH-USDT"}}
    return strategy


class TestExecutorFactoryRegistry(unittest.TestCase):
    """Tests that verify the registry state without creating executor instances."""

    def test_all_executors_registered(self):
        """All 8 config types must be in the registry."""
        registry = ExecutorFactory.get_registry()
        for config_cls in EXPECTED_REGISTRY:
            self.assertIn(
                config_cls,
                registry,
                f"{config_cls.__name__} not found in ExecutorFactory registry",
            )

    def test_registry_returns_correct_classes(self):
        """get_registry() must map each config type to its executor class."""
        registry = ExecutorFactory.get_registry()
        for config_cls, executor_cls in EXPECTED_REGISTRY.items():
            self.assertIs(
                registry[config_cls],
                executor_cls,
                f"Expected {executor_cls.__name__} for {config_cls.__name__}, got {registry[config_cls].__name__}",
            )

    def test_registry_has_the_eight_core_entries(self):
        """Registry must contain all 8 factory-native executor types (>= 8 total allowed for DAG safety)."""
        registry = ExecutorFactory.get_registry()
        for config_cls in EXPECTED_REGISTRY:
            self.assertIn(
                config_cls,
                registry,
                f"{config_cls.__name__} not found in ExecutorFactory registry",
            )
        self.assertGreaterEqual(
            len(registry),
            8,
            f"Expected at least 8 entries, got {len(registry)}: {list(registry.keys())}",
        )

    def test_is_registered_true_for_known_types(self):
        """is_registered() returns True for all known config types."""
        for config_cls in EXPECTED_REGISTRY:
            self.assertTrue(
                ExecutorFactory.is_registered(config_cls),
                f"is_registered({config_cls.__name__}) returned False",
            )

    def test_is_registered_false_for_unknown_type(self):
        """is_registered() returns False for an unregistered config type."""

        class FakeConfig(ExecutorConfigBase):
            type: str = "fake_executor"

        self.assertFalse(ExecutorFactory.is_registered(FakeConfig))

    def test_create_unknown_config_raises(self):
        """Factory.create() raises ValueError for unregistered config type."""

        class UnknownConfig(ExecutorConfigBase):
            type: str = "unknown_executor"

        config = UnknownConfig()
        strategy = _make_mock_strategy()
        with self.assertRaises(ValueError) as ctx:
            ExecutorFactory.create(strategy=strategy, config=config)
        self.assertIn("No executor registered", str(ctx.exception))
        self.assertIn("UnknownConfig", str(ctx.exception))


class TestExecutorFactoryCreate(unittest.TestCase):
    """Tests that verify factory creates correct executor instances.

    Each test patches executor.start() to prevent event loop issues and
    validates the returned instance type.
    """

    def setUp(self):
        self.strategy = _make_mock_strategy()

    @patch.object(PositionExecutor, "start")
    def test_create_position_executor(self, _start_mock):
        config = PositionExecutorConfig(
            timestamp=1234,
            connector_name="binance",
            trading_pair="ETH-USDT",
            side=TradeType.BUY,
            entry_price=Decimal(100),
            amount=Decimal(10),
        )
        executor = ExecutorFactory.create(strategy=self.strategy, config=config)
        self.assertIsInstance(executor, PositionExecutor)
        self.assertIs(executor.config, config)

    @patch.object(DCAExecutor, "start")
    def test_create_dca_executor(self, _start_mock):
        config = DCAExecutorConfig(
            timestamp=1234,
            connector_name="binance",
            trading_pair="ETH-USDT",
            side=TradeType.BUY,
            amounts_quote=[Decimal(10)],
            prices=[Decimal(100)],
        )
        executor = ExecutorFactory.create(strategy=self.strategy, config=config)
        self.assertIsInstance(executor, DCAExecutor)
        self.assertIs(executor.config, config)

    @patch.object(GridExecutor, "start")
    @patch.object(GridExecutor, "_generate_grid_levels")
    def test_create_grid_executor(self, _grid_levels_mock, _start_mock):
        config = GridExecutorConfig(
            timestamp=1234,
            connector_name="binance",
            trading_pair="ETH-USDT",
            side=TradeType.BUY,
            total_amount_quote=Decimal(100),
            start_price=Decimal(100),
            end_price=Decimal(200),
            limit_price=Decimal(90),
            triple_barrier_config=TripleBarrierConfig(take_profit=Decimal("0.01"), stop_loss=Decimal("0.2")),
        )
        executor = ExecutorFactory.create(strategy=self.strategy, config=config)
        self.assertIsInstance(executor, GridExecutor)
        self.assertIs(executor.config, config)

    @patch.object(TWAPExecutor, "start")
    def test_create_twap_executor(self, _start_mock):
        config = TWAPExecutorConfig(
            timestamp=1234,
            connector_name="binance",
            trading_pair="ETH-USDT",
            side=TradeType.BUY,
            total_amount_quote=Decimal(100),
            total_duration=10,
            order_interval=5,
        )
        executor = ExecutorFactory.create(strategy=self.strategy, config=config)
        self.assertIsInstance(executor, TWAPExecutor)
        self.assertIs(executor.config, config)

    @patch.object(OrderExecutor, "start")
    def test_create_order_executor(self, _start_mock):
        config = OrderExecutorConfig(
            timestamp=1234,
            connector_name="binance",
            trading_pair="ETH-USDT",
            side=TradeType.BUY,
            amount=Decimal(10),
            price=Decimal(100),
            execution_strategy=ExecutionStrategy.LIMIT,
        )
        executor = ExecutorFactory.create(strategy=self.strategy, config=config)
        self.assertIsInstance(executor, OrderExecutor)
        self.assertIs(executor.config, config)

    @patch.object(ArbitrageExecutor, "start")
    def test_create_arbitrage_executor(self, _start_mock):
        # Arbitrage needs two connectors in strategy
        connector2 = MagicMock(spec=ExchangePyBase)
        type(connector2).trading_rules = PropertyMock(return_value={"ETH-USDT": TradingRule(trading_pair="ETH-USDT")})
        self.strategy.connectors["mock_exchange_2"] = connector2
        config = ArbitrageExecutorConfig(
            timestamp=1234,
            order_amount=Decimal(10),
            min_profitability=Decimal("0.01"),
            buying_market=ConnectorPair(connector_name="binance", trading_pair="ETH-USDT"),
            selling_market=ConnectorPair(connector_name="mock_exchange_2", trading_pair="ETH-USDT"),
        )
        executor = ExecutorFactory.create(strategy=self.strategy, config=config)
        self.assertIsInstance(executor, ArbitrageExecutor)

    @patch.object(XEMMExecutor, "start")
    def test_create_xemm_executor(self, _start_mock):
        # XEMM needs two connectors in strategy; use generic mock names to avoid
        # real connector validation (e.g. "coinbase does not support market orders")
        connector1 = MagicMock(spec=ExchangePyBase)
        type(connector1).trading_rules = PropertyMock(return_value={"ETH-USDT": TradingRule(trading_pair="ETH-USDT")})
        connector1.supported_order_types.return_value = [OrderType.MARKET, OrderType.LIMIT, OrderType.LIMIT_MAKER]
        connector2 = MagicMock(spec=ExchangePyBase)
        type(connector2).trading_rules = PropertyMock(return_value={"ETH-USDT": TradingRule(trading_pair="ETH-USDT")})
        connector2.supported_order_types.return_value = [OrderType.MARKET, OrderType.LIMIT, OrderType.LIMIT_MAKER]
        self.strategy.connectors["mock_exchange_1"] = connector1
        self.strategy.connectors["mock_exchange_2"] = connector2
        config = XEMMExecutorConfig(
            timestamp=1234,
            buying_market=ConnectorPair(connector_name="mock_exchange_1", trading_pair="ETH-USDT"),
            selling_market=ConnectorPair(connector_name="mock_exchange_2", trading_pair="ETH-USDT"),
            maker_side=TradeType.BUY,
            order_amount=Decimal(10),
            min_profitability=Decimal("0.01"),
            target_profitability=Decimal("0.02"),
            max_profitability=Decimal("0.05"),
        )
        executor = ExecutorFactory.create(strategy=self.strategy, config=config)
        self.assertIsInstance(executor, XEMMExecutor)

    @patch.object(LPExecutor, "start")
    def test_create_lp_executor(self, _start_mock):
        config = LPExecutorConfig(
            timestamp=1234,
            connector_name="binance",
            trading_pair="ETH-USDT",
            lp_provider="meteora/clmm",
            side=TradeType.BUY,
            pool_address="0xabc123",
            lower_price=Decimal(100),
            upper_price=Decimal(200),
            base_amount=Decimal(1),
            quote_amount=Decimal(100),
        )
        executor = ExecutorFactory.create(strategy=self.strategy, config=config)
        self.assertIsInstance(executor, LPExecutor)
        self.assertIs(executor.config, config)


class TestExecutorFactoryLogger(unittest.TestCase):
    """Tests for the ExecutorFactory logger lazy initialization and warning paths."""

    def setUp(self):
        # Reset _logger so each test starts fresh
        ExecutorFactory._logger = None

    def tearDown(self):
        # Restore after test
        ExecutorFactory._logger = None

    def test_factory_logger_initialization(self):
        """logger() lazily creates and caches the logger instance."""
        self.assertIsNone(ExecutorFactory._logger)
        logger1 = ExecutorFactory.logger()
        self.assertIsNotNone(logger1)
        # Second call must return the exact same cached object
        logger2 = ExecutorFactory.logger()
        self.assertIs(logger1, logger2)

    def test_factory_duplicate_registration_warning(self):
        """Registering a second executor for the same config type logs a warning."""
        import logging

        class DuplicateConfig(ExecutorConfigBase):
            type: str = "duplicate_executor"

        class FirstExecutor:
            pass

        class SecondExecutor:
            pass

        # Seed registry with the first mapping
        ExecutorFactory._registry[DuplicateConfig] = FirstExecutor
        try:
            with self.assertLogs("hummingbot.strategy_v2.executors.executor_factory", level=logging.WARNING) as cm:
                ExecutorFactory.register(DuplicateConfig)(SecondExecutor)
            self.assertTrue(
                any("Overriding executor registration" in msg for msg in cm.output),
                f"Expected override warning in log output, got: {cm.output}",
            )
            # Registry must now point to the replacement class
            self.assertIs(ExecutorFactory._registry[DuplicateConfig], SecondExecutor)
        finally:
            del ExecutorFactory._registry[DuplicateConfig]


class TestOrchestratorNoneControllerId(unittest.TestCase):
    """Tests for orchestrator behavior when controller_id is None."""

    @patch("hummingbot.strategy_v2.executors.executor_orchestrator.MarketsRecorder.get_instance")
    def test_orchestrator_action_with_none_controller_id(self, markets_recorder_mock):
        """execute_action() with controller_id=None logs an error and returns without crashing."""
        import logging

        from hummingbot.strategy_v2.executors.executor_orchestrator import ExecutorOrchestrator
        from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction

        markets_recorder_mock.return_value = MagicMock()
        markets_recorder_mock.return_value.get_all_executors = MagicMock(return_value=[])
        markets_recorder_mock.return_value.get_all_positions = MagicMock(return_value=[])

        strategy = _make_mock_strategy()
        orchestrator = ExecutorOrchestrator(strategy=strategy)

        config = DCAExecutorConfig(
            timestamp=1234,
            connector_name="binance",
            trading_pair="ETH-USDT",
            side=TradeType.BUY,
            amounts_quote=[Decimal(10)],
            prices=[Decimal(100)],
        )
        # Force controller_id to None after construction (field default is "main")
        action = CreateExecutorAction(executor_config=config, controller_id="main")
        action.controller_id = None

        # Should not raise — error is logged and method returns early
        with self.assertLogs("hummingbot.strategy_v2.executors.executor_orchestrator", level=logging.ERROR) as cm:
            orchestrator.execute_action(action)

        self.assertTrue(
            any("controller_id=None" in msg for msg in cm.output),
            f"Expected controller_id=None error log, got: {cm.output}",
        )
        # No executors should have been created
        self.assertEqual(orchestrator.active_executors, {})


class TestOrchestratorLegacyFallback(unittest.TestCase):
    """Tests for the legacy string-keyed _executor_mapping fallback in create_executor."""

    @patch("hummingbot.strategy_v2.executors.executor_orchestrator.MarketsRecorder.get_instance")
    def test_orchestrator_legacy_fallback(self, markets_recorder_mock):
        """When factory raises ValueError, orchestrator falls back to _executor_mapping."""
        from hummingbot.strategy_v2.executors.executor_orchestrator import ExecutorOrchestrator
        from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction

        markets_recorder_mock.return_value = MagicMock()
        markets_recorder_mock.return_value.get_all_executors = MagicMock(return_value=[])
        markets_recorder_mock.return_value.get_all_positions = MagicMock(return_value=[])

        strategy = _make_mock_strategy()
        orchestrator = ExecutorOrchestrator(strategy=strategy)

        mock_executor = MagicMock()
        mock_executor.config = MagicMock()
        mock_executor.config.id = "legacy-id"
        mock_executor.config.controller_id = "ctrl"

        MockExecutorClass = MagicMock(return_value=mock_executor)

        # Temporarily clear factory registry so factory raises ValueError
        original_registry = dict(ExecutorFactory._registry)
        ExecutorFactory._registry.clear()
        try:
            orchestrator._executor_mapping = {"legacy_type": MockExecutorClass}

            class LegacyConfig(ExecutorConfigBase):
                type: str = "legacy_type"

            config = LegacyConfig()
            action = CreateExecutorAction(executor_config=config, controller_id="ctrl")
            orchestrator.execute_action(action)

            # Legacy class must have been instantiated
            MockExecutorClass.assert_called_once()
            self.assertIn("ctrl", orchestrator.active_executors)
            self.assertIn(mock_executor, orchestrator.active_executors["ctrl"])
        finally:
            ExecutorFactory._registry.update(original_registry)

    @patch("hummingbot.strategy_v2.executors.executor_orchestrator.MarketsRecorder.get_instance")
    def test_orchestrator_legacy_fallback_raises_for_unknown(self, markets_recorder_mock):
        """When factory and _executor_mapping both lack the type, ValueError is raised."""
        from hummingbot.strategy_v2.executors.executor_orchestrator import ExecutorOrchestrator
        from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction

        markets_recorder_mock.return_value = MagicMock()
        markets_recorder_mock.return_value.get_all_executors = MagicMock(return_value=[])
        markets_recorder_mock.return_value.get_all_positions = MagicMock(return_value=[])

        strategy = _make_mock_strategy()
        orchestrator = ExecutorOrchestrator(strategy=strategy)

        original_registry = dict(ExecutorFactory._registry)
        ExecutorFactory._registry.clear()
        try:
            # Empty legacy mapping too
            orchestrator._executor_mapping = {}

            class TotallyUnknownConfig(ExecutorConfigBase):
                type: str = "totally_unknown"

            config = TotallyUnknownConfig()
            action = CreateExecutorAction(executor_config=config, controller_id="ctrl")

            with self.assertRaises(ValueError) as ctx:
                orchestrator.execute_action(action)

            self.assertIn("No executor registered", str(ctx.exception))
        finally:
            ExecutorFactory._registry.update(original_registry)


class TestOrchestratorUsesFactory(unittest.TestCase):
    """Verify the orchestrator delegates to ExecutorFactory.create()."""

    @patch("hummingbot.strategy_v2.executors.executor_orchestrator.ExecutorFactory.create")
    @patch("hummingbot.strategy_v2.executors.executor_orchestrator.MarketsRecorder.get_instance")
    def test_orchestrator_uses_factory(self, markets_recorder_mock, factory_create_mock):
        from hummingbot.strategy_v2.executors.executor_orchestrator import ExecutorOrchestrator
        from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction

        markets_recorder_mock.return_value = MagicMock()
        markets_recorder_mock.return_value.get_all_executors = MagicMock(return_value=[])
        markets_recorder_mock.return_value.get_all_positions = MagicMock(return_value=[])

        strategy = _make_mock_strategy()
        orchestrator = ExecutorOrchestrator(strategy=strategy)

        # Create a mock executor that the factory will return
        mock_executor = MagicMock()
        mock_executor.config = MagicMock()
        mock_executor.config.id = "test-id"
        mock_executor.config.controller_id = "test"
        factory_create_mock.return_value = mock_executor

        config = DCAExecutorConfig(
            timestamp=1234,
            connector_name="binance",
            trading_pair="ETH-USDT",
            side=TradeType.BUY,
            amounts_quote=[Decimal(10)],
            prices=[Decimal(100)],
        )
        action = CreateExecutorAction(executor_config=config, controller_id="test")
        orchestrator.execute_actions([action])

        factory_create_mock.assert_called_once()
        call_kwargs = factory_create_mock.call_args.kwargs
        self.assertEqual(call_kwargs["strategy"], strategy)
        self.assertIsInstance(call_kwargs["config"], DCAExecutorConfig)


class TestOrchestratorUpdateCachedPerformance(unittest.TestCase):
    """Tests for _update_cached_performance covering CloseType branch paths."""

    def setUp(self):
        from hummingbot.strategy_v2.executors.executor_orchestrator import ExecutorOrchestrator

        mr_patcher = patch("hummingbot.strategy_v2.executors.executor_orchestrator.MarketsRecorder.get_instance")
        mr_mock = mr_patcher.start()
        self.addCleanup(mr_patcher.stop)
        mr_mock.return_value = MagicMock()
        mr_mock.return_value.get_all_executors = MagicMock(return_value=[])
        mr_mock.return_value.get_all_positions = MagicMock(return_value=[])

        self.strategy = _make_mock_strategy()
        self.orchestrator = ExecutorOrchestrator(strategy=self.strategy)

    def _make_executor_info(self, close_type, net_pnl_quote=Decimal("5"), filled_amount_quote=Decimal("100")):
        from hummingbot.strategy_v2.models.executors_info import ExecutorInfo

        info = MagicMock(spec=ExecutorInfo)
        info.close_type = close_type
        info.net_pnl_quote = net_pnl_quote
        info.filled_amount_quote = filled_amount_quote
        return info

    def _fresh_report(self):
        from hummingbot.strategy_v2.models.executors_info import PerformanceReport

        return PerformanceReport()

    def test_update_cached_performance_non_position_hold_adds_pnl(self):
        """Lines 217-219: close_type != POSITION_HOLD adds to realized_pnl and volume."""
        from hummingbot.strategy_v2.models.executors import CloseType

        report = self._fresh_report()
        self.orchestrator.cached_performance["ctrl"] = report

        info = self._make_executor_info(close_type=CloseType.STOP_LOSS)
        self.orchestrator._update_cached_performance("ctrl", info)

        self.assertEqual(report.realized_pnl_quote, Decimal("5"))
        self.assertEqual(report.volume_traded, Decimal("100"))

    def test_update_cached_performance_position_hold_skips_pnl(self):
        """Line 217 false-branch: close_type == POSITION_HOLD skips realized_pnl update."""
        from hummingbot.strategy_v2.models.executors import CloseType

        report = self._fresh_report()
        self.orchestrator.cached_performance["ctrl"] = report

        info = self._make_executor_info(close_type=CloseType.POSITION_HOLD)
        self.orchestrator._update_cached_performance("ctrl", info)

        self.assertEqual(report.realized_pnl_quote, Decimal("0"))
        self.assertEqual(report.volume_traded, Decimal("0"))

    def test_update_cached_performance_close_type_counts_incremented(self):
        """Lines 221-222: close_type truthy increments close_type_counts."""
        from hummingbot.strategy_v2.models.executors import CloseType

        report = self._fresh_report()
        self.orchestrator.cached_performance["ctrl"] = report

        info = self._make_executor_info(close_type=CloseType.TIME_LIMIT)
        self.orchestrator._update_cached_performance("ctrl", info)
        self.assertEqual(report.close_type_counts[CloseType.TIME_LIMIT], 1)

        # Second call increments existing count
        self.orchestrator._update_cached_performance("ctrl", info)
        self.assertEqual(report.close_type_counts[CloseType.TIME_LIMIT], 2)

    def test_update_cached_performance_none_close_type_skips_counts(self):
        """Line 220 false-branch: close_type=None leaves close_type_counts empty."""
        report = self._fresh_report()
        self.orchestrator.cached_performance["ctrl"] = report

        info = self._make_executor_info(close_type=None)
        self.orchestrator._update_cached_performance("ctrl", info)

        self.assertEqual(report.close_type_counts, {})

    def test_update_cached_performance_creates_report_for_new_controller(self):
        """Lines 213-214: new controller_id creates a fresh PerformanceReport."""
        from hummingbot.strategy_v2.models.executors import CloseType
        from hummingbot.strategy_v2.models.executors_info import PerformanceReport

        self.assertNotIn("new_ctrl", self.orchestrator.cached_performance)
        info = self._make_executor_info(close_type=CloseType.STOP_LOSS)
        self.orchestrator._update_cached_performance("new_ctrl", info)
        self.assertIn("new_ctrl", self.orchestrator.cached_performance)
        self.assertIsInstance(self.orchestrator.cached_performance["new_ctrl"], PerformanceReport)


if __name__ == "__main__":
    unittest.main()
