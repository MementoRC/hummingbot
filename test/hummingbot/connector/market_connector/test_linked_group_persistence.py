import asyncio
from decimal import Decimal
import json
import tempfile
from types import SimpleNamespace
import unittest

import pytest

pytest.importorskip("market_connector")
pytest.importorskip("strategy_framework")
pytest.importorskip("state_store")

from data_type_primitives import (  # noqa: E402
    LegRole,
    LegSpec,
    LinkedOrderSpec,
    LinkKind,
    OrderState,
    TradeType,
)
from data_type_primitives.common import OrderType  # noqa: E402
from market_connector.exchanges.kraken.linked_order_codec import KrakenClOrdIdLinkCodec  # noqa: E402
from market_connector.protocols import LegSnapshot as McLegSnapshot  # noqa: E402
from state_store.sqlite import SqliteStateStore  # noqa: E402
from strategy_framework.linked_orders import (  # noqa: E402
    EmulatedLinkedOrderManager,
    GroupIdPersistenceError,
    GroupIdSpaceExhaustedError,
    LinkedOrderManagerConfig,
)

from hummingbot.connector.market_connector.kraken_emulation_venue import KrakenEmulationVenue  # noqa: E402
from hummingbot.connector.market_connector.linked_group_persistence import (  # noqa: E402
    FleetGroupIdAllocator,
    SlotConflictError,
    check_slot_conflict,
    open_linked_group_store,
    recover_with_slot_check,
)

from .fakes import PAIR, FakeGateway  # noqa: E402

D = Decimal
MAX_ID = KrakenClOrdIdLinkCodec.GROUP_ID_MAX


def _oco_spec():
    sell = TradeType.SELL
    return LinkedOrderSpec(
        kind=LinkKind.OCO,
        legs=(
            LegSpec(role=LegRole.OCO_LEG, order_type=OrderType.LIMIT, trade_type=sell, amount=D("1"), price=D("110")),
            LegSpec(
                role=LegRole.OCO_LEG,
                order_type=OrderType.LIMIT,
                trade_type=sell,
                amount=D("1.5"),
                price=D("120.25"),
            ),
        ),
    )


def _open_leg(tag):
    return SimpleNamespace(client_order_id=tag, link_tag=tag, state=OrderState.OPEN, filled=D(0))


def _manager(gateway, state_store, group_store, host_slot=0, account_key="acct"):
    codec = KrakenClOrdIdLinkCodec()
    venue = KrakenEmulationVenue(gateway, codec)
    allocator = FleetGroupIdAllocator(state_store, account_key=account_key, host_slot=host_slot, max_group_id=MAX_ID)
    config = LinkedOrderManagerConfig(enabled=True, require_recover_before_place=True, cancel_orphans=False)
    manager = EmulatedLinkedOrderManager(venue, codec, config, store=group_store, allocator=allocator)
    return manager, venue, codec


class _TmpFileCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = f"{self._tmp.name}/linked.sqlite"


class SerializerTests(_TmpFileCase):
    async def test_group_round_trips_through_store(self):
        gateway = FakeGateway()
        state_store, group_store = open_linked_group_store(self.path)
        self.addCleanup(state_store.close)
        manager, _, _ = _manager(gateway, state_store, group_store)
        await manager.recover([PAIR])
        await manager.place_oco(PAIR, _oco_spec())

        saved = group_store.load_all()
        assert len(saved) == 1
        record = saved[0]
        assert record.spec == _oco_spec()
        assert record.spec.legs[1].price == D("120.25")
        # Stored form is plain JSON (Decimal as str), and a second store on the file reads the same record.
        raw = state_store.records.load("linked_groups", str(record.group_id))
        json.dumps(raw)
        _, other_store = open_linked_group_store(self.path)
        assert other_store.load_all() == [record]
        assert group_store.durable is True


class RestartRecoveryTests(_TmpFileCase):
    async def test_new_manager_on_same_file_recovers_group(self):
        gateway = FakeGateway()
        state_store, group_store = open_linked_group_store(self.path)
        manager, _, _ = _manager(gateway, state_store, group_store)
        await manager.recover([PAIR])
        handle = await manager.place_oco(PAIR, _oco_spec())
        tags = handle.leg_client_order_ids
        await manager.shutdown()
        state_store.close()

        gateway2 = FakeGateway()
        gateway2.open_legs = [McLegSnapshot(t, t, OrderState.OPEN, D(0)) for t in tags]
        gateway2.get_leg_return = McLegSnapshot(tags[0], tags[0], OrderState.OPEN, D(0))
        state_store2, group_store2 = open_linked_group_store(self.path)
        self.addCleanup(state_store2.close)
        manager2, venue2, codec2 = _manager(gateway2, state_store2, group_store2)
        await recover_with_slot_check(manager2, venue2, codec2, group_store2, [PAIR], 0)
        assert [r.group_id for r in group_store2.load_all()] == [codec2.decode(tags[0])[0]]
        assert codec2.decode(tags[0])[0] in manager2._groups  # private attr: no public observable
        # The next id continues past the persisted counter: it is never reused after a restart.
        handle2 = await manager2.place_oco(PAIR, _oco_spec())
        assert codec2.decode(handle2.leg_client_order_ids[0])[0] > codec2.decode(tags[0])[0]


class AllocatorTests(_TmpFileCase):
    def _allocator(self, state_store, **kw):
        params = {"account_key": "acct", "host_slot": 0, "max_group_id": MAX_ID}
        params.update(kw)
        return FleetGroupIdAllocator(state_store, **params)

    async def test_ids_are_in_slot_and_monotonic(self):
        state_store = SqliteStateStore(self.path)
        self.addCleanup(state_store.close)
        alloc = self._allocator(state_store, host_slot=3)
        ids = [await alloc.reserve_group_id() for _ in range(3)]
        assert ids == [3 * 10**9 + 1, 3 * 10**9 + 2, 3 * 10**9 + 3]

    async def test_interleaved_allocators_never_duplicate(self):
        store_a = SqliteStateStore(self.path)
        store_b = SqliteStateStore(self.path)  # second instance = separate connection
        self.addCleanup(store_a.close)
        self.addCleanup(store_b.close)
        allocs = [self._allocator(store_a), self._allocator(store_b), self._allocator(store_a)]
        results = await asyncio.gather(*(allocs[i % 3].reserve_group_id() for i in range(60)))
        assert len(set(results)) == 60

    async def test_counter_survives_restart(self):
        first = SqliteStateStore(self.path)
        first_id = await self._allocator(first).reserve_group_id()
        first.close()
        second = SqliteStateStore(self.path)
        self.addCleanup(second.close)
        assert await self._allocator(second).reserve_group_id() == first_id + 1

    async def test_accounts_and_slots_have_separate_counters(self):
        state_store = SqliteStateStore(self.path)
        self.addCleanup(state_store.close)
        a = await self._allocator(state_store, account_key="one").reserve_group_id()
        b = await self._allocator(state_store, account_key="two").reserve_group_id()
        assert a == b == 1
        assert "one" not in self._allocator(state_store, account_key="one")._name.replace("linked_group_id", "")

    def test_slot_bounds_validated(self):
        state_store = SqliteStateStore(self.path)
        self.addCleanup(state_store.close)
        for bad in (-1, 100, True, "1"):
            with pytest.raises(ValueError):
                self._allocator(state_store, host_slot=bad)
        self._allocator(state_store, host_slot=99, max_group_id=99_999_999_999)
        with pytest.raises(ValueError):
            self._allocator(state_store, host_slot=99, max_group_id=99_999_999_998)

    async def test_overflow_fails_closed_with_no_venue_call(self):
        gateway = FakeGateway()
        state_store, group_store = open_linked_group_store(self.path)
        self.addCleanup(state_store.close)
        manager, _, _ = _manager(gateway, state_store, group_store)
        allocator = manager._allocator
        state_store.counters.advance_to(allocator._name, 10**9 - 1)
        with pytest.raises(GroupIdSpaceExhaustedError):
            await allocator.reserve_group_id()
        await manager.recover([PAIR])
        with pytest.raises(GroupIdPersistenceError):
            await manager.place_oco(PAIR, _oco_spec())
        assert [c for c in gateway.calls if c[0] == "place_order"] == []
        assert group_store.load_all() == []


class SlotConflictTests(_TmpFileCase):
    async def test_foreign_group_in_our_slot_refuses(self):
        gateway = FakeGateway()
        codec = KrakenClOrdIdLinkCodec()
        gateway.open_legs = [_open_leg(codec.encode(7, 0))]
        state_store, group_store = open_linked_group_store(self.path)
        self.addCleanup(state_store.close)
        manager, venue, _ = _manager(gateway, state_store, group_store)
        with pytest.raises(SlotConflictError):
            await recover_with_slot_check(manager, venue, codec, group_store, [PAIR], 0)
        with pytest.raises(Exception):  # noqa: B017 - recover never completed, so placing is refused
            await manager.place_oco(PAIR, _oco_spec())
        assert [c for c in gateway.calls if c[0] == "place_order"] == []

    async def test_other_slot_and_foreign_tags_are_fine(self):
        gateway = FakeGateway()
        codec = KrakenClOrdIdLinkCodec()
        gateway.open_legs = [_open_leg(codec.encode(10**9 + 4, 0)), _open_leg("hbot-buy-1")]
        state_store, group_store = open_linked_group_store(self.path)
        self.addCleanup(state_store.close)
        manager, venue, _ = _manager(gateway, state_store, group_store)
        await check_slot_conflict(venue, codec, group_store, [PAIR], 0)

    async def test_known_group_in_our_slot_is_fine(self):
        gateway = FakeGateway()
        state_store, group_store = open_linked_group_store(self.path)
        self.addCleanup(state_store.close)
        manager, venue, codec = _manager(gateway, state_store, group_store)
        await manager.recover([PAIR])
        handle = await manager.place_oco(PAIR, _oco_spec())
        gateway.open_legs = [_open_leg(tag) for tag in handle.leg_client_order_ids]
        await check_slot_conflict(venue, codec, group_store, [PAIR], 0)
