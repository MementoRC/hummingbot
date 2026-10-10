"""Shared retry logic for executors.

Tracks retry count and evaluates max retries threshold.
Executors increment retries at their own sites (order failed, shutdown loop).
"""

from __future__ import annotations


class RetryMixin:
    """Mixin providing retry count tracking and max-retries evaluation.

    Usage:
        class MyExecutor(RetryMixin, ExecutorBase):
            def __init__(self, ..., max_retries=10):
                super().__init__(...)
                self.init_retry(max_retries)

            async def control_task(self):
                ...
                self.evaluate_max_retries()

            def process_order_failed_event(self, ...):
                ...
                self.increment_retries("order failed")
    """

    def init_retry(self, max_retries: int = 10) -> None:
        """Initialize retry state. Call from __init__ after super().__init__."""
        self._current_retries: int = 0
        self._max_retries: int = max_retries

    @property
    def current_retries(self) -> int:
        return self._current_retries

    @current_retries.setter
    def current_retries(self, value: int) -> None:
        self._current_retries = value

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def increment_retries(self, reason: str = "") -> None:
        """Increment retry counter."""
        self._current_retries += 1

    def reset_retries(self) -> None:
        """Reset retry counter to 0."""
        self._current_retries = 0

    def evaluate_max_retries(self) -> None:
        """Stop executor if max retries exceeded.

        Uses > (not >=) for consistency across all executors.
        Auto-detects stop method: close_execution_by() if available, else stop().
        """
        if self._current_retries > self._max_retries:
            from hummingbot.strategy_v2.models.executors import CloseType

            if hasattr(self, "close_execution_by"):
                self.close_execution_by(CloseType.FAILED)
            else:
                self.close_type = CloseType.FAILED
                self.stop()
