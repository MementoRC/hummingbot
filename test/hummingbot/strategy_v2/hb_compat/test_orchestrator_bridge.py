"""Tests for OrchestratorBridge — thin hummingbot ↔ strategy-framework bridge."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

strategy_framework = pytest.importorskip("strategy_framework", reason="strategy-framework sub-package not installed")
from strategy_framework.hb_compat import OrchestratorAdapter  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_bridge(**kwargs):
    """Construct an OrchestratorBridge with minimal valid arguments."""
    from hummingbot.strategy_v2.hb_compat import OrchestratorBridge

    market_access = MagicMock(name="market_access")
    market_data = MagicMock(name="market_data")
    with patch.object(OrchestratorBridge, "__init__", wraps=OrchestratorBridge.__init__):
        bridge = OrchestratorBridge.__new__(OrchestratorBridge)
        # Bypass StrategyV2Base.__init__ — we only test the bridge layer
        bridge._adapter = OrchestratorAdapter(market_access, market_data)
    return bridge, market_data


# ── Construction ──────────────────────────────────────────────────────────────


class TestOrchestratorBridgeConstruction:
    def test_holds_orchestrator_adapter(self):
        bridge, _ = _make_bridge()
        assert isinstance(bridge.adapter, OrchestratorAdapter)

    def test_adapter_property_same_instance(self):
        bridge, _ = _make_bridge()
        assert bridge.adapter is bridge.adapter


# ── on_tick delegates to adapter.evaluate ─────────────────────────────────────


class TestOnTick:
    def test_on_tick_calls_adapter_evaluate(self):
        bridge, market_data = _make_bridge()
        controller = MagicMock()
        controller.controller_id = "c1"
        controller.evaluate.return_value = []
        bridge.register_controller(controller)

        bridge.on_tick()

        controller.evaluate.assert_called_once_with(market_data)

    def test_on_tick_multiple_times(self):
        bridge, market_data = _make_bridge()
        controller = MagicMock()
        controller.controller_id = "c1"
        controller.evaluate.return_value = []
        bridge.register_controller(controller)

        bridge.on_tick()
        bridge.on_tick()

        assert controller.evaluate.call_count == 2


# ── Controller management ─────────────────────────────────────────────────────


class TestControllerManagement:
    def test_register_and_evaluate(self):
        bridge, market_data = _make_bridge()
        controller = MagicMock()
        controller.controller_id = "ctrl1"
        controller.evaluate.return_value = []

        bridge.register_controller(controller)
        bridge.on_tick()

        controller.evaluate.assert_called_once()

    def test_unregister_stops_evaluation(self):
        bridge, _ = _make_bridge()
        controller = MagicMock()
        controller.controller_id = "ctrl1"
        controller.evaluate.return_value = []

        bridge.register_controller(controller)
        bridge.unregister_controller("ctrl1")
        bridge.on_tick()

        controller.evaluate.assert_not_called()


# ── Delegation ────────────────────────────────────────────────────────────────


class TestDelegation:
    def test_get_active_executors_empty_by_default(self):
        bridge, _ = _make_bridge()
        assert bridge.get_active_executors("no-such") == []

    def test_get_executor_state_unknown_raises(self):
        bridge, _ = _make_bridge()
        with pytest.raises(KeyError):
            bridge.get_executor_state("no-such")
