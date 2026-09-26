"""Tests for ShutdownMixin."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from hummingbot.strategy_v2.executors.mixins.shutdown import ShutdownMixin


class MockShutdownExecutor(ShutdownMixin):
    """Mock executor implementing ShutdownMixin template methods."""

    def __init__(
        self,
        open_filled=Decimal("100"),
        close_filled=Decimal("100"),
        min_order_size=Decimal("1"),
        has_pending=False,
    ):
        self._open_filled = open_filled
        self._close_filled = close_filled
        self._min_size = min_order_size
        self._has_pending = has_pending
        self.stop = MagicMock()
        self.increment_retries = MagicMock()
        self._place_called = False
        self._update_called = False

    def _get_open_filled_amount(self) -> Decimal:
        return self._open_filled

    def _get_close_filled_amount(self) -> Decimal:
        return self._close_filled

    def _get_min_order_size(self) -> Decimal:
        return self._min_size

    def _has_pending_close_order(self) -> bool:
        return self._has_pending

    async def _place_shutdown_close_order(self) -> None:
        self._place_called = True

    async def _update_pending_close_order(self) -> None:
        self._update_called = True


class TestShutdownMixin:
    @pytest.mark.asyncio
    async def test_flat_position_stops(self):
        """When open == close, executor should stop."""
        executor = MockShutdownExecutor(
            open_filled=Decimal("100"),
            close_filled=Decimal("100"),
            min_order_size=Decimal("1"),
        )
        await executor.control_shutdown_process()
        executor.stop.assert_called_once()
        assert not executor._place_called
        assert not executor._update_called

    @pytest.mark.asyncio
    async def test_nearly_flat_position_stops(self):
        """When difference < min_order_size, executor should stop."""
        executor = MockShutdownExecutor(
            open_filled=Decimal("100"),
            close_filled=Decimal("99.5"),
            min_order_size=Decimal("1"),
        )
        await executor.control_shutdown_process()
        executor.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_pending_close_order_updates(self):
        """When close order is pending, should update not place new."""
        executor = MockShutdownExecutor(
            open_filled=Decimal("100"),
            close_filled=Decimal("0"),
            min_order_size=Decimal("1"),
            has_pending=True,
        )
        await executor.control_shutdown_process()
        assert executor._update_called
        assert not executor._place_called
        executor.stop.assert_not_called()
        executor.increment_retries.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_pending_order_places_close(self):
        """When no close order pending, should place one and increment retries."""
        executor = MockShutdownExecutor(
            open_filled=Decimal("100"),
            close_filled=Decimal("0"),
            min_order_size=Decimal("1"),
            has_pending=False,
        )
        await executor.control_shutdown_process()
        assert executor._place_called
        assert not executor._update_called
        executor.increment_retries.assert_called_once_with("shutdown close order")
        executor.stop.assert_not_called()

    def test_not_implemented_hooks(self):
        """Base ShutdownMixin raises NotImplementedError for required hooks."""
        mixin = ShutdownMixin()
        with pytest.raises(NotImplementedError):
            mixin._get_open_filled_amount()
        with pytest.raises(NotImplementedError):
            mixin._get_close_filled_amount()
        with pytest.raises(NotImplementedError):
            mixin._has_pending_close_order()

    @pytest.mark.asyncio
    async def test_not_implemented_place_order(self):
        mixin = ShutdownMixin()
        with pytest.raises(NotImplementedError):
            await mixin._place_shutdown_close_order()

    def test_default_min_order_size(self):
        """Default min_order_size is a small epsilon."""
        mixin = ShutdownMixin()
        assert mixin._get_min_order_size() == Decimal("1e-10")

    def test_default_sleep_interval(self):
        mixin = ShutdownMixin()
        assert mixin._get_shutdown_sleep_interval() == 5.0

    @pytest.mark.asyncio
    async def test_custom_on_shutdown_flat(self):
        """Subclass can override _on_shutdown_flat."""

        class CustomShutdown(MockShutdownExecutor):
            def __init__(self):
                super().__init__(open_filled=Decimal("0"), close_filled=Decimal("0"))
                self.custom_flat_called = False

            def _on_shutdown_flat(self):
                self.custom_flat_called = True

        executor = CustomShutdown()
        await executor.control_shutdown_process()
        assert executor.custom_flat_called
        executor.stop.assert_not_called()
