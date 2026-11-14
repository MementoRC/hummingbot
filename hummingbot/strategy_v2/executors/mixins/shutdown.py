"""Shared shutdown logic for executors.

Provides the common control_shutdown_process() pattern:
1. Check if open_filled ~= close_filled -> stop()
2. If close order pending -> wait/update
3. Else -> place close order, increment retries
4. Sleep
"""

from __future__ import annotations

import asyncio
from decimal import Decimal


class ShutdownMixin:
    """Mixin providing standardized shutdown process.

    Handles the common pattern of ensuring positions are flat
    before stopping. Executors override the template methods
    to plug in their specific order tracking.

    Usage:
        class MyExecutor(ShutdownMixin, ExecutorBase):
            def _get_open_filled_amount(self) -> Decimal: ...
            def _get_close_filled_amount(self) -> Decimal: ...
            def _get_min_order_size(self) -> Decimal: ...
            def _has_pending_close_order(self) -> bool: ...
            async def _place_shutdown_close_order(self) -> None: ...
            async def _update_pending_close_order(self) -> None: ...
    """

    async def control_shutdown_process(self) -> None:
        """Standard shutdown: ensure position is flat, retry close orders."""
        open_filled = self._get_open_filled_amount()
        close_filled = self._get_close_filled_amount()

        if abs(open_filled - close_filled) < self._get_min_order_size():
            self._on_shutdown_flat()
            return

        if self._has_pending_close_order():
            await self._update_pending_close_order()
        else:
            await self._place_shutdown_close_order()
            self.increment_retries("shutdown close order")

        await asyncio.sleep(self._get_shutdown_sleep_interval())

    def _get_open_filled_amount(self) -> Decimal:
        """Override to return total open filled amount."""
        raise NotImplementedError

    def _get_close_filled_amount(self) -> Decimal:
        """Override to return total close filled amount."""
        raise NotImplementedError

    def _get_min_order_size(self) -> Decimal:
        """Override to return minimum order size for flat-position check.

        Default uses a small epsilon for float comparison.
        """
        return Decimal("1e-10")

    def _has_pending_close_order(self) -> bool:
        """Override to check if a close order is already pending."""
        raise NotImplementedError

    async def _place_shutdown_close_order(self) -> None:
        """Override to place the close order during shutdown."""
        raise NotImplementedError

    async def _update_pending_close_order(self) -> None:
        """Override to update/monitor pending close orders."""
        await asyncio.sleep(1.0)

    def _on_shutdown_flat(self) -> None:
        """Called when position is flat during shutdown. Default: stop()."""
        self.stop()

    def _get_shutdown_sleep_interval(self) -> float:
        """Override to customize the sleep interval during shutdown."""
        return 5.0
