import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import pytest

pytest.importorskip("market_connector")
pytest.importorskip("strategy_framework")

from market_connector.exchanges.kraken.linked_order_codec import KrakenClOrdIdLinkCodec  # noqa: E402
from strategy_framework.linked_orders import EmulatedLinkedOrderManager  # noqa: E402

from hummingbot.connector.exchange.kraken_mc.kraken_mc_exchange import KrakenMcExchange  # noqa: E402
from hummingbot.connector.market_connector.kraken_emulation_venue import KrakenEmulationVenue  # noqa: E402
from hummingbot.connector.market_connector.market_connector_exchange import MarketConnectorExchange  # noqa: E402

from .fakes import PAIR, FakeDescriptor, FakeGateway  # noqa: E402


class _KrakenLikeExchange(MarketConnectorExchange):
    """Same hook override as KrakenMcExchange, without building the real Kraken descriptor."""

    LINKED_TICK_INTERVAL_S = 0.01

    def _build_linked_venue(self, handle):
        codec = KrakenClOrdIdLinkCodec()
        return KrakenEmulationVenue(handle.gateway, codec), codec


def _make(cls):
    descriptor = FakeDescriptor()
    exchange = cls(
        descriptor=descriptor,
        config={},
        connector_name="fake_mc",
        trading_pairs=[PAIR],
        order_book_refresh_interval=10,
    )
    return exchange, descriptor


class LinkedWiringTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.exchange, self.descriptor = _make(_KrakenLikeExchange)
        self.gateway = self.descriptor.handle.gateway
        self.events = []

    async def asyncTearDown(self):
        await self.exchange.stop_network()

    def _record_tick(self):
        async def tick(manager):
            self.events.append("tick")

        return patch.object(EmulatedLinkedOrderManager, "tick", tick)

    async def test_manager_built_on_start_network(self):
        assert self.exchange.linked_order_manager is None
        await self.exchange.start_network()
        assert isinstance(self.exchange.linked_order_manager, EmulatedLinkedOrderManager)

    def test_kraken_mc_overrides_hook(self):
        assert KrakenMcExchange._build_linked_venue is not MarketConnectorExchange._build_linked_venue
        venue, codec = KrakenMcExchange._build_linked_venue(SimpleNamespace(), SimpleNamespace(gateway=FakeGateway()))
        assert isinstance(venue, KrakenEmulationVenue)
        assert isinstance(codec, KrakenClOrdIdLinkCodec)

    async def test_stop_network_twice_is_noop(self):
        await self.exchange.start_network()
        await self.exchange.stop_network()
        await self.exchange.stop_network()
        assert self.descriptor.handle.closed == 1
        assert self.gateway.unsubscribed == 1

    async def test_stop_network_before_start_is_noop(self):
        await self.exchange.stop_network()
        assert self.descriptor.handle.closed == 0
        assert self.exchange._handle is None

    async def test_stop_network_shutdown_failure_still_closes_handle(self):
        async def shutdown(manager):
            raise RuntimeError("shutdown boom")

        with patch.object(EmulatedLinkedOrderManager, "shutdown", shutdown):
            await self.exchange.start_network()
            refresh_task = self.exchange._refresh_task
            with pytest.raises(RuntimeError, match="shutdown boom"):
                await self.exchange.stop_network()
        assert self.descriptor.handle.closed == 1
        assert refresh_task.cancelled()
        assert self.exchange._handle is None
        assert self.exchange._refresh_task is None

    async def test_no_manager_for_non_kraken_descriptor(self):
        exchange, descriptor = _make(MarketConnectorExchange)
        await exchange.start_network()
        try:
            assert exchange.linked_order_manager is None
            assert exchange._linked_tick_task is None
            assert descriptor.handle.gateway.calls == []
        finally:
            await exchange.stop_network()

    async def test_recover_runs_before_place(self):
        order = []
        original_recover = EmulatedLinkedOrderManager.recover

        async def recover(manager, targets):
            order.append(("recover", list(targets), self.gateway.callbacks != []))
            await original_recover(manager, targets)

        async def tick(manager):
            order.append(("tick",))

        with (
            patch.object(EmulatedLinkedOrderManager, "recover", recover),
            patch.object(EmulatedLinkedOrderManager, "tick", tick),
        ):
            await self.exchange.start_network()
            await asyncio.sleep(0.05)
        # Subscribed before recover ran, recover covers this connector's pairs, and ticks come after it.
        assert order[0] == ("recover", [PAIR], True)
        assert ("tick",) in order[1:]
        # Placing is refused until recover() has completed: it did, so the flag is set.
        assert self.exchange.linked_order_manager._recovered is True  # private attr: no public observable

    async def test_tick_task_runs(self):
        with self._record_tick():
            await self.exchange.start_network()
            await asyncio.sleep(0.1)
            assert self.events.count("tick") >= 2
            assert not self.exchange._linked_tick_task.done()

    async def test_stop_calls_shutdown_and_unsubscribe(self):
        calls = []

        async def shutdown(manager):
            calls.append("shutdown")

        with patch.object(EmulatedLinkedOrderManager, "shutdown", shutdown):
            await self.exchange.start_network()
            tick_task = self.exchange._linked_tick_task
            await self.exchange.stop_network()
        assert calls == ["shutdown"]
        assert self.gateway.unsubscribed == 1
        assert tick_task.done()
        assert self.exchange.linked_order_manager is None
        assert self.exchange._linked_tick_task is None
        assert self.exchange._leg_fill_unsub is None

    async def test_start_network_recover_failure_closes_handle(self):
        async def recover(manager, targets):
            raise RuntimeError("recover boom")

        with patch.object(EmulatedLinkedOrderManager, "recover", recover):
            with pytest.raises(RuntimeError, match="recover boom"):
                await self.exchange.start_network()
        assert self.descriptor.handle.closed == 1
        assert self.gateway.unsubscribed == 1
        assert self.exchange.linked_order_manager is None
        assert self.exchange._linked_tick_task is None
        assert self.exchange._handle is None
