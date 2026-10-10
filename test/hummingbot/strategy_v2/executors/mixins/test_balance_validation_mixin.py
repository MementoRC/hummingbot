"""Tests for BalanceValidationMixin.

Note: We mock the hummingbot.strategy_v2.models.executors module to avoid
the Cython import chain (limit_order) in worktrees without built extensions.
The mixin does a deferred import of CloseType inside validate_sufficient_balance().
"""

from decimal import Decimal
from enum import Enum
import sys
from unittest.mock import MagicMock, patch

import pytest

from hummingbot.strategy_v2.executors.mixins.balance_validation import BalanceValidationMixin


# Create a mock CloseType enum for tests
class MockCloseType(Enum):
    INSUFFICIENT_BALANCE = "insufficient_balance"
    FAILED = "failed"


# Pre-populate sys.modules with a mock for the executors module if the real
# one can't be imported (Cython not built). This lets the deferred import
# inside the mixin succeed.
_mock_executors_module = None
try:
    from hummingbot.strategy_v2.models.executors import CloseType  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _mock_executors_module = MagicMock()
    _mock_executors_module.CloseType = MockCloseType
    sys.modules.setdefault("hummingbot.strategy_v2.models.executors", _mock_executors_module)
    # Also ensure intermediate modules exist
    for mod_name in [
        "hummingbot.core.data_type.in_flight_order",
        "hummingbot.core.data_type.limit_order",
    ]:
        sys.modules.setdefault(mod_name, MagicMock())
    CloseType = MockCloseType


class MockOrderCandidate:
    """Minimal OrderCandidate-like object for testing."""

    def __init__(self, amount: Decimal):
        self.amount = amount


class MockExecutor(BalanceValidationMixin):
    """Mock executor with BalanceValidationMixin."""

    def __init__(self, candidate_amount=Decimal("100"), adjusted_amount=Decimal("100")):
        self._candidate_amount = candidate_amount
        self._adjusted_amount = adjusted_amount
        self.close_type = None
        self.stop = MagicMock()
        self.config = MagicMock()
        self.config.connector_name = "binance"

    def _create_validation_order_candidate(self):
        return MockOrderCandidate(amount=self._candidate_amount)

    def adjust_order_candidates(self, exchange, candidates):
        return [MockOrderCandidate(amount=self._adjusted_amount)]

    def logger(self):
        return MagicMock()


class TestBalanceValidationMixin:
    @pytest.mark.asyncio
    async def test_sufficient_balance_does_not_stop(self):
        executor = MockExecutor(adjusted_amount=Decimal("100"))
        await executor.validate_sufficient_balance()
        executor.stop.assert_not_called()
        assert executor.close_type is None

    @pytest.mark.asyncio
    async def test_insufficient_balance_stops_executor(self):
        executor = MockExecutor(adjusted_amount=Decimal("0"))
        await executor.validate_sufficient_balance()
        executor.stop.assert_called_once()
        assert executor.close_type == CloseType.INSUFFICIENT_BALANCE

    @pytest.mark.asyncio
    async def test_calls_create_validation_order_candidate(self):
        executor = MockExecutor(adjusted_amount=Decimal("50"))
        with patch.object(
            executor, "_create_validation_order_candidate", wraps=executor._create_validation_order_candidate
        ) as mock_create:
            await executor.validate_sufficient_balance()
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_calls_adjust_order_candidates_with_connector(self):
        executor = MockExecutor(adjusted_amount=Decimal("50"))
        with patch.object(executor, "adjust_order_candidates", wraps=executor.adjust_order_candidates) as mock_adjust:
            await executor.validate_sufficient_balance()
            mock_adjust.assert_called_once()
            args = mock_adjust.call_args
            assert args[0][0] == "binance"

    def test_not_implemented_without_override(self):
        """Base mixin raises NotImplementedError for _create_validation_order_candidate."""
        mixin = BalanceValidationMixin()
        with pytest.raises(NotImplementedError):
            mixin._create_validation_order_candidate()

    @pytest.mark.asyncio
    async def test_nonzero_adjusted_amount_passes(self):
        """Any non-zero adjusted amount means sufficient balance."""
        executor = MockExecutor(adjusted_amount=Decimal("0.001"))
        await executor.validate_sufficient_balance()
        executor.stop.assert_not_called()
        assert executor.close_type is None
