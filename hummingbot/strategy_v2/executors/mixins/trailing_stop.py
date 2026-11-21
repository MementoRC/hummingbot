"""Shared trailing stop ratchet logic for executors."""

from __future__ import annotations

from decimal import Decimal


class TrailingStopMixin:
    """Mixin providing the trailing stop ratchet algorithm.

    The algorithm:
    1. Wait until PNL exceeds activation_price
    2. Set trigger = pnl - trailing_delta
    3. Ratchet trigger upward as PNL rises
    4. Fire when PNL drops below trigger

    Usage:
        class MyExecutor(TrailingStopMixin, ExecutorBase):
            def __init__(self, ...):
                super().__init__(...)
                self.init_trailing_stop()

            def control_trailing_stop(self):
                if self.evaluate_trailing_stop():
                    self.place_close_order_and_cancel_open_orders(
                        close_type=CloseType.TRAILING_STOP)

            def _get_trailing_stop_pnl_pct(self):
                return self.get_net_pnl_pct()

            def _get_trailing_stop_config(self):
                return self.config.trailing_stop
    """

    def init_trailing_stop(self) -> None:
        """Initialize trailing stop state. Call from __init__ after super().__init__."""
        self._trailing_stop_trigger_pct: Decimal | None = None

    def evaluate_trailing_stop(self) -> bool:
        """Evaluate trailing stop condition.

        Returns True if the trailing stop should fire.
        Returns False if not activated, not yet triggered, or no config.
        """
        config = self._get_trailing_stop_config()
        if not config:
            return False

        pnl_pct = self._get_trailing_stop_pnl_pct()

        if not self._trailing_stop_trigger_pct:
            # Not yet activated
            if pnl_pct > config.activation_price:
                self._trailing_stop_trigger_pct = pnl_pct - config.trailing_delta
            return False

        # Already activated — check fire condition
        if pnl_pct < self._trailing_stop_trigger_pct:
            return True

        # Ratchet trigger upward
        if pnl_pct - config.trailing_delta > self._trailing_stop_trigger_pct:
            self._trailing_stop_trigger_pct = pnl_pct - config.trailing_delta

        return False

    def _get_trailing_stop_pnl_pct(self) -> Decimal:
        """Override: return the PNL percentage to track."""
        raise NotImplementedError

    def _get_trailing_stop_config(self):
        """Override: return config with .activation_price and .trailing_delta, or None."""
        raise NotImplementedError
