"""Integration tests: verify mixins work with real executor classes.

These tests instantiate real PositionExecutor, DCAExecutor, and TWAPExecutor
with a mocked strategy, confirming that mixin methods are present, callable,
and correctly dispatch through the MRO.
"""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.dca_executor.data_types import DCAExecutorConfig
from hummingbot.strategy_v2.executors.dca_executor.dca_executor import DCAExecutor
from hummingbot.strategy_v2.executors.mixins.activation_bounds import ActivationBoundsMixin
from hummingbot.strategy_v2.executors.mixins.balance_validation import BalanceValidationMixin
from hummingbot.strategy_v2.executors.mixins.order_tracking import OrderTrackingMixin
from hummingbot.strategy_v2.executors.mixins.pnl_calculator import PNLCalculatorMixin
from hummingbot.strategy_v2.executors.mixins.retry import RetryMixin
from hummingbot.strategy_v2.executors.mixins.trailing_stop import TrailingStopMixin
from hummingbot.strategy_v2.executors.position_executor.data_types import (
    PositionExecutorConfig,
    TrailingStop,
    TripleBarrierConfig,
)
from hummingbot.strategy_v2.executors.position_executor.position_executor import PositionExecutor
from hummingbot.strategy_v2.executors.twap_executor.data_types import TWAPExecutorConfig
from hummingbot.strategy_v2.executors.twap_executor.twap_executor import TWAPExecutor
from hummingbot.strategy_v2.models.executors import CloseType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_trading_rule() -> MagicMock:
    """Create a mock TradingRule to avoid Cython import."""
    rule = MagicMock()
    rule.trading_pair = "BTC-USDT"
    rule.min_order_size = Decimal("0.001")
    rule.min_price_increment = Decimal("0.01")
    rule.min_base_amount_increment = Decimal("0.001")
    rule.min_notional_size = Decimal("10")
    return rule


def _mock_connector(mid_price: Decimal = Decimal("100")) -> MagicMock:
    """Create a mock connector with trading rules and price lookup."""
    connector = MagicMock()
    connector.get_price_by_type.return_value = mid_price
    connector.trading_rules = {"BTC-USDT": _mock_trading_rule()}
    connector.budget_checker.adjust_candidates.side_effect = lambda candidates, all_or_none: candidates
    return connector


def _mock_strategy(mid_price: Decimal = Decimal("100")) -> MagicMock:
    """Create a mock strategy with a single connector."""
    strategy = MagicMock(spec=StrategyV2Base)
    connector = _mock_connector(mid_price)
    strategy.connectors = {"binance_paper_trade": connector}
    strategy.current_timestamp = 1_000_000.0
    return strategy


@pytest.fixture
def strategy():
    return _mock_strategy()


@pytest.fixture
def position_executor(strategy):
    config = PositionExecutorConfig(
        connector_name="binance_paper_trade",
        trading_pair="BTC-USDT",
        side=TradeType.BUY,
        amount=Decimal("0.1"),
        triple_barrier_config=TripleBarrierConfig(
            stop_loss=Decimal("0.05"),
            take_profit=Decimal("0.1"),
            trailing_stop=TrailingStop(
                activation_price=Decimal("0.03"),
                trailing_delta=Decimal("0.01"),
            ),
        ),
        activation_bounds=[Decimal("0.01"), Decimal("0.02")],
    )
    return PositionExecutor(strategy=strategy, config=config, max_retries=5)


@pytest.fixture
def dca_executor(strategy):
    config = DCAExecutorConfig(
        connector_name="binance_paper_trade",
        trading_pair="BTC-USDT",
        side=TradeType.BUY,
        amounts_quote=[Decimal("50"), Decimal("50")],
        prices=[Decimal("100"), Decimal("95")],
        trailing_stop=TrailingStop(
            activation_price=Decimal("0.03"),
            trailing_delta=Decimal("0.01"),
        ),
    )
    return DCAExecutor(strategy=strategy, config=config, max_retries=15)


@pytest.fixture
def twap_executor(strategy):
    config = TWAPExecutorConfig(
        connector_name="binance_paper_trade",
        trading_pair="BTC-USDT",
        side=TradeType.BUY,
        total_amount_quote=Decimal("100"),
        total_duration=60,
        order_interval=20,
    )
    return TWAPExecutor(strategy=strategy, config=config, max_retries=15)


# ---------------------------------------------------------------------------
# 1. PositionExecutor has RetryMixin
# ---------------------------------------------------------------------------


class TestPositionExecutorRetryMixin:
    def test_isinstance_check(self, position_executor):
        assert isinstance(position_executor, RetryMixin)

    def test_init_retry_sets_state(self, position_executor):
        assert position_executor.current_retries == 0
        assert position_executor.max_retries == 5

    def test_increment_and_evaluate(self, position_executor):
        for _ in range(5):
            position_executor.increment_retries("test")
        assert position_executor.current_retries == 5
        # At limit (5 == 5), should NOT stop (uses >)
        position_executor.evaluate_max_retries()
        assert position_executor.status.name != "TERMINATED"


# ---------------------------------------------------------------------------
# 2. PositionExecutor has BalanceValidationMixin
# ---------------------------------------------------------------------------


class TestPositionExecutorBalanceValidation:
    def test_isinstance_check(self, position_executor):
        assert isinstance(position_executor, BalanceValidationMixin)

    def test_create_validation_order_candidate_exists(self, position_executor):
        assert hasattr(position_executor, "_create_validation_order_candidate")
        assert callable(position_executor._create_validation_order_candidate)


# ---------------------------------------------------------------------------
# 3. PositionExecutor has TrailingStopMixin
# ---------------------------------------------------------------------------


class TestPositionExecutorTrailingStop:
    def test_isinstance_check(self, position_executor):
        assert isinstance(position_executor, TrailingStopMixin)

    def test_evaluate_trailing_stop_exists(self, position_executor):
        assert hasattr(position_executor, "evaluate_trailing_stop")
        assert callable(position_executor.evaluate_trailing_stop)

    def test_init_trailing_stop_ran(self, position_executor):
        """init_trailing_stop should have set _trailing_stop_trigger_pct to None."""
        assert position_executor._trailing_stop_trigger_pct is None


# ---------------------------------------------------------------------------
# 4. PositionExecutor has ActivationBoundsMixin
# ---------------------------------------------------------------------------


class TestPositionExecutorActivationBounds:
    def test_isinstance_check(self, position_executor):
        assert isinstance(position_executor, ActivationBoundsMixin)

    def test_is_within_activation_bounds_callable(self, position_executor):
        assert callable(position_executor._is_within_activation_bounds)

    def test_within_bounds_returns_bool(self, position_executor):
        result = position_executor._is_within_activation_bounds(
            order_price=Decimal("100"),
            side=TradeType.BUY,
            order_type=OrderType.MARKET,
        )
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 5. DCAExecutor has RetryMixin
# ---------------------------------------------------------------------------


class TestDCAExecutorRetryMixin:
    def test_isinstance_check(self, dca_executor):
        assert isinstance(dca_executor, RetryMixin)

    def test_retry_methods_accessible(self, dca_executor):
        assert dca_executor.current_retries == 0
        assert dca_executor.max_retries == 15
        dca_executor.increment_retries("test")
        assert dca_executor.current_retries == 1
        dca_executor.reset_retries()
        assert dca_executor.current_retries == 0


# ---------------------------------------------------------------------------
# 6. DCAExecutor has OrderTrackingMixin
# ---------------------------------------------------------------------------


class TestDCAExecutorOrderTracking:
    def test_isinstance_check(self, dca_executor):
        assert isinstance(dca_executor, OrderTrackingMixin)

    def test_get_trackable_orders_returns_list(self, dca_executor):
        orders = dca_executor._get_trackable_orders()
        assert isinstance(orders, list)


# ---------------------------------------------------------------------------
# 7. TWAPExecutor has PNLCalculatorMixin
# ---------------------------------------------------------------------------


class TestTWAPExecutorPNLCalculator:
    def test_isinstance_check(self, twap_executor):
        assert isinstance(twap_executor, PNLCalculatorMixin)

    def test_trade_pnl_pct_returns_decimal(self, twap_executor):
        """With no fills, entry price is 0, so trade_pnl_pct returns 0."""
        result = twap_executor.trade_pnl_pct
        assert isinstance(result, Decimal)
        assert result == Decimal("0")


# ---------------------------------------------------------------------------
# 8. MRO puts mixins before ExecutorBase
# ---------------------------------------------------------------------------


class TestMixinMROOrder:
    def test_position_executor_mro(self):
        mro = PositionExecutor.__mro__
        mro_names = [cls.__name__ for cls in mro]
        # All mixins should appear before ExecutorBase in MRO
        exec_base_idx = mro_names.index("ExecutorBase")
        for mixin_name in ["TrailingStopMixin", "ActivationBoundsMixin", "RetryMixin", "BalanceValidationMixin"]:
            mixin_idx = mro_names.index(mixin_name)
            assert mixin_idx < exec_base_idx, (
                f"{mixin_name} (idx={mixin_idx}) should come before ExecutorBase (idx={exec_base_idx}) in MRO"
            )

    def test_dca_executor_mro(self):
        mro = DCAExecutor.__mro__
        mro_names = [cls.__name__ for cls in mro]
        exec_base_idx = mro_names.index("ExecutorBase")
        for mixin_name in ["PNLCalculatorMixin", "TrailingStopMixin", "OrderTrackingMixin", "RetryMixin"]:
            mixin_idx = mro_names.index(mixin_name)
            assert mixin_idx < exec_base_idx, (
                f"{mixin_name} (idx={mixin_idx}) should come before ExecutorBase (idx={exec_base_idx}) in MRO"
            )

    def test_twap_executor_mro(self):
        mro = TWAPExecutor.__mro__
        mro_names = [cls.__name__ for cls in mro]
        exec_base_idx = mro_names.index("ExecutorBase")
        for mixin_name in ["PNLCalculatorMixin", "OrderTrackingMixin", "RetryMixin", "BalanceValidationMixin"]:
            mixin_idx = mro_names.index(mixin_name)
            assert mixin_idx < exec_base_idx, (
                f"{mixin_name} (idx={mixin_idx}) should come before ExecutorBase (idx={exec_base_idx}) in MRO"
            )


# ---------------------------------------------------------------------------
# 9. evaluate_max_retries uses close_execution_by (DCA)
# ---------------------------------------------------------------------------


class TestEvaluateMaxRetriesUsesCloseExecutionBy:
    def test_dca_uses_close_execution_by(self, dca_executor):
        """DCA has close_execution_by, so evaluate_max_retries should use it."""
        assert hasattr(dca_executor, "close_execution_by")
        dca_executor.close_execution_by = MagicMock()
        # Push past max_retries
        for _ in range(16):
            dca_executor.increment_retries("test")
        dca_executor.evaluate_max_retries()
        dca_executor.close_execution_by.assert_called_once_with(CloseType.FAILED)


# ---------------------------------------------------------------------------
# 10. evaluate_max_retries uses stop() (Position)
# ---------------------------------------------------------------------------


class TestEvaluateMaxRetriesUsesStop:
    def test_position_uses_stop(self, position_executor):
        """PositionExecutor does NOT have close_execution_by, so it falls back to stop()."""
        assert not hasattr(PositionExecutor, "close_execution_by") or not callable(
            getattr(PositionExecutor, "close_execution_by", None)
        )
        position_executor.stop = MagicMock()
        # Push past max_retries (5)
        for _ in range(6):
            position_executor.increment_retries("test")
        position_executor.evaluate_max_retries()
        position_executor.stop.assert_called_once()
        assert position_executor.close_type == CloseType.FAILED
