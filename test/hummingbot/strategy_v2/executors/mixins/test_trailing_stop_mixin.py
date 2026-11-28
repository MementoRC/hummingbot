"""Tests for TrailingStopMixin."""

from decimal import Decimal

from hummingbot.strategy_v2.executors.mixins.trailing_stop import TrailingStopMixin


class MockTrailingStopConfig:
    def __init__(self, activation_price, trailing_delta):
        self.activation_price = activation_price
        self.trailing_delta = trailing_delta


class MockExecutor(TrailingStopMixin):
    """Mock executor implementing trailing stop template methods."""

    def __init__(self, pnl_pct=Decimal("0"), config=None):
        self.init_trailing_stop()
        self._pnl_pct = pnl_pct
        self._ts_config = config

    def _get_trailing_stop_pnl_pct(self):
        return self._pnl_pct

    def _get_trailing_stop_config(self):
        return self._ts_config


class TestTrailingStopMixin:
    def test_init_trailing_stop(self):
        executor = MockExecutor()
        assert executor._trailing_stop_trigger_pct is None

    def test_no_config_returns_false(self):
        executor = MockExecutor(pnl_pct=Decimal("0.05"), config=None)
        assert executor.evaluate_trailing_stop() is False

    def test_below_activation_no_trigger(self):
        config = MockTrailingStopConfig(
            activation_price=Decimal("0.03"),
            trailing_delta=Decimal("0.01"),
        )
        executor = MockExecutor(pnl_pct=Decimal("0.02"), config=config)
        assert executor.evaluate_trailing_stop() is False
        assert executor._trailing_stop_trigger_pct is None

    def test_activation_sets_trigger(self):
        config = MockTrailingStopConfig(
            activation_price=Decimal("0.03"),
            trailing_delta=Decimal("0.01"),
        )
        executor = MockExecutor(pnl_pct=Decimal("0.05"), config=config)
        result = executor.evaluate_trailing_stop()
        assert result is False
        assert executor._trailing_stop_trigger_pct == Decimal("0.04")  # 0.05 - 0.01

    def test_ratchet_upward(self):
        config = MockTrailingStopConfig(
            activation_price=Decimal("0.03"),
            trailing_delta=Decimal("0.01"),
        )
        executor = MockExecutor(pnl_pct=Decimal("0.05"), config=config)
        executor.evaluate_trailing_stop()  # activate at trigger=0.04
        assert executor._trailing_stop_trigger_pct == Decimal("0.04")

        # PNL rises to 0.08 -> new trigger should be 0.07
        executor._pnl_pct = Decimal("0.08")
        result = executor.evaluate_trailing_stop()
        assert result is False
        assert executor._trailing_stop_trigger_pct == Decimal("0.07")

    def test_no_ratchet_downward(self):
        config = MockTrailingStopConfig(
            activation_price=Decimal("0.03"),
            trailing_delta=Decimal("0.01"),
        )
        executor = MockExecutor(pnl_pct=Decimal("0.08"), config=config)
        executor.evaluate_trailing_stop()  # activate at trigger=0.07

        # PNL drops to 0.075 -- still above trigger, but new_trigger=0.065 < current 0.07
        executor._pnl_pct = Decimal("0.075")
        result = executor.evaluate_trailing_stop()
        assert result is False
        assert executor._trailing_stop_trigger_pct == Decimal("0.07")  # unchanged

    def test_fire_when_below_trigger(self):
        config = MockTrailingStopConfig(
            activation_price=Decimal("0.03"),
            trailing_delta=Decimal("0.01"),
        )
        executor = MockExecutor(pnl_pct=Decimal("0.05"), config=config)
        executor.evaluate_trailing_stop()  # activate at trigger=0.04

        # PNL drops below trigger
        executor._pnl_pct = Decimal("0.035")
        result = executor.evaluate_trailing_stop()
        assert result is True

    def test_exact_activation_boundary_no_trigger(self):
        """At exactly activation_price, should NOT activate (uses >)."""
        config = MockTrailingStopConfig(
            activation_price=Decimal("0.03"),
            trailing_delta=Decimal("0.01"),
        )
        executor = MockExecutor(pnl_pct=Decimal("0.03"), config=config)
        result = executor.evaluate_trailing_stop()
        assert result is False
        assert executor._trailing_stop_trigger_pct is None

    def test_exact_trigger_boundary_no_fire(self):
        """At exactly trigger value, should NOT fire (uses <)."""
        config = MockTrailingStopConfig(
            activation_price=Decimal("0.03"),
            trailing_delta=Decimal("0.01"),
        )
        executor = MockExecutor(pnl_pct=Decimal("0.05"), config=config)
        executor.evaluate_trailing_stop()  # trigger=0.04

        executor._pnl_pct = Decimal("0.04")
        result = executor.evaluate_trailing_stop()
        assert result is False

    def test_multiple_ratchet_steps(self):
        config = MockTrailingStopConfig(
            activation_price=Decimal("0.01"),
            trailing_delta=Decimal("0.005"),
        )
        executor = MockExecutor(pnl_pct=Decimal("0.02"), config=config)
        executor.evaluate_trailing_stop()  # trigger=0.015
        assert executor._trailing_stop_trigger_pct == Decimal("0.015")

        executor._pnl_pct = Decimal("0.03")
        executor.evaluate_trailing_stop()  # trigger=0.025
        assert executor._trailing_stop_trigger_pct == Decimal("0.025")

        executor._pnl_pct = Decimal("0.05")
        executor.evaluate_trailing_stop()  # trigger=0.045
        assert executor._trailing_stop_trigger_pct == Decimal("0.045")

        # Now drop below
        executor._pnl_pct = Decimal("0.04")
        result = executor.evaluate_trailing_stop()
        assert result is True
