"""Tests for RetryMixin."""

from unittest.mock import MagicMock

from hummingbot.strategy_v2.executors.mixins.retry import RetryMixin


class MockExecutorWithCloseExecution(RetryMixin):
    """Mock executor that has close_execution_by."""

    def __init__(self, max_retries=10):
        self.init_retry(max_retries)
        self.close_execution_by = MagicMock()


class MockExecutorWithStop(RetryMixin):
    """Mock executor that only has stop()."""

    def __init__(self, max_retries=10):
        self.init_retry(max_retries)
        self.close_type = None
        self.stop = MagicMock()


class TestRetryMixin:
    def test_init_retry_defaults(self):
        executor = MockExecutorWithStop()
        assert executor.current_retries == 0
        assert executor.max_retries == 10

    def test_init_retry_custom(self):
        executor = MockExecutorWithStop(max_retries=25)
        assert executor.max_retries == 25

    def test_increment_retries(self):
        executor = MockExecutorWithStop()
        executor.increment_retries("test reason")
        assert executor.current_retries == 1
        executor.increment_retries()
        assert executor.current_retries == 2

    def test_reset_retries(self):
        executor = MockExecutorWithStop()
        executor.increment_retries()
        executor.increment_retries()
        assert executor.current_retries == 2
        executor.reset_retries()
        assert executor.current_retries == 0

    def test_property_setter(self):
        executor = MockExecutorWithStop()
        executor.current_retries = 5
        assert executor.current_retries == 5

    def test_evaluate_under_limit_no_stop(self):
        executor = MockExecutorWithStop(max_retries=5)
        for _ in range(5):
            executor.increment_retries()
        executor.evaluate_max_retries()
        executor.stop.assert_not_called()

    def test_evaluate_at_limit_no_stop(self):
        """At exactly max_retries, should NOT stop (uses > not >=)."""
        executor = MockExecutorWithStop(max_retries=3)
        for _ in range(3):
            executor.increment_retries()
        executor.evaluate_max_retries()
        executor.stop.assert_not_called()

    def test_evaluate_over_limit_triggers_stop(self):
        """Over max_retries triggers stop."""
        executor = MockExecutorWithStop(max_retries=3)
        for _ in range(4):
            executor.increment_retries()
        executor.evaluate_max_retries()
        executor.stop.assert_called_once()
        from hummingbot.strategy_v2.models.executors import CloseType

        assert executor.close_type == CloseType.FAILED

    def test_evaluate_uses_close_execution_by_when_available(self):
        executor = MockExecutorWithCloseExecution(max_retries=1)
        executor.increment_retries()
        executor.increment_retries()
        executor.evaluate_max_retries()
        from hummingbot.strategy_v2.models.executors import CloseType

        executor.close_execution_by.assert_called_once_with(CloseType.FAILED)

    def test_evaluate_zero_max_retries(self):
        """With max_retries=0, first increment triggers stop."""
        executor = MockExecutorWithStop(max_retries=0)
        executor.increment_retries()
        executor.evaluate_max_retries()
        executor.stop.assert_called_once()
