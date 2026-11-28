"""Shared balance validation for executors.

Standardizes the validate_sufficient_balance() pattern used by most
executors: build an OrderCandidate, run it through the budget checker,
stop if the adjusted amount is zero.
"""

from __future__ import annotations

from decimal import Decimal


class BalanceValidationMixin:
    """Mixin for validating sufficient balance before executor start.

    Subclasses implement _create_validation_order_candidate() to build the
    appropriate OrderCandidate (or PerpetualOrderCandidate) for their
    specific order type and configuration.

    Usage:
        class MyExecutor(BalanceValidationMixin, ExecutorBase):
            def _create_validation_order_candidate(self):
                return OrderCandidate(...)

    Note: DCA executor is excluded because it validates multiple
    candidates (one per level) with different stop semantics.
    """

    async def validate_sufficient_balance(self) -> None:
        """Check balance via budget checker and stop if insufficient."""
        from hummingbot.strategy_v2.models.executors import CloseType

        order_candidate = self._create_validation_order_candidate()
        adjusted = self.adjust_order_candidates(self.config.connector_name, [order_candidate])
        if adjusted[0].amount == Decimal("0"):
            self.close_type = CloseType.INSUFFICIENT_BALANCE
            self.logger().error("Not enough budget to open position.")
            self.stop()

    def _create_validation_order_candidate(self):
        """Override to create the order candidate for balance validation.

        Must return an OrderCandidate or PerpetualOrderCandidate instance.
        """
        raise NotImplementedError
