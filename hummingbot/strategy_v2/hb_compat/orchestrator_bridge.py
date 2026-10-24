"""OrchestratorBridge — thin hummingbot bridge to OrchestratorAdapter.

Inherits StrategyV2Base so the hummingbot Clock drives evaluation, then
delegates every call to the protocol-based OrchestratorAdapter which has
no hummingbot dependency.

Usage::

    access = LiveMarketAccess(connector, "BTC-USDT")
    data   = LiveMarketData(market_data_provider, "binance")
    bridge = OrchestratorBridge(connectors, access, data)
    bridge.register_controller(my_controller)
    # hummingbot Clock calls bridge.on_tick() each tick
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from strategy_framework.hb_compat import OrchestratorAdapter

from hummingbot.strategy.strategy_v2_base import StrategyV2Base

if TYPE_CHECKING:
    from pydantic import BaseModel
    from strategy_framework.primitives.enums import RunnableStatus
    from strategy_framework.protocols import MarketAccessProtocol, MarketDataProtocol
    from strategy_framework.protocols.event_bus import EventBusProtocol

    from hummingbot.connector.connector_base import ConnectorBase


class OrchestratorBridge(StrategyV2Base):
    """Bridges hummingbot's Clock-driven on_tick() to OrchestratorAdapter.evaluate().

    This class is intentionally thin: it owns no strategy logic. All
    controller management and action dispatch are delegated to the adapter.
    """

    def __init__(
        self,
        connectors: dict[str, ConnectorBase],
        market_access: MarketAccessProtocol,
        market_data: MarketDataProtocol,
        config: BaseModel | None = None,
        event_bus: EventBusProtocol | None = None,
    ) -> None:
        super().__init__(connectors, config)
        self._adapter = OrchestratorAdapter(market_access, market_data, event_bus)

    # ── Core tick ─────────────────────────────────────────────────────────────

    def on_tick(self) -> None:
        """Drive the adapter one evaluation cycle per Clock tick."""
        self._adapter.evaluate()

    # ── Controller management ─────────────────────────────────────────────────

    def register_controller(self, controller: Any) -> None:
        self._adapter.register_controller(controller)

    def unregister_controller(self, controller_id: str) -> None:
        self._adapter.unregister_controller(controller_id)

    # ── OrchestratorProtocol delegation ──────────────────────────────────────

    def submit_actions(self, actions: list[Any]) -> None:
        self._adapter.submit_actions(actions)

    def get_executor_state(self, executor_id: str) -> RunnableStatus:
        return self._adapter.get_executor_state(executor_id)

    def get_active_executors(self, controller_id: str) -> list[str]:
        return self._adapter.get_active_executors(controller_id)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def adapter(self) -> OrchestratorAdapter:
        """The underlying OrchestratorAdapter instance."""
        return self._adapter
