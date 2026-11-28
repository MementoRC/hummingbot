"""Durable linked-order persistence for market-connector exchanges (hummingbot#85 P4b).

Built on hb-state-store: a group store over a ``SqliteStateStore`` record collection and a multi-host
group-id allocator over its durable counters. The record codec is the strategy-framework one
(``group_record_to_dict`` / ``group_record_from_dict``): versioned, JSON-safe and round-trip exact.
"""

import asyncio
import hashlib
import logging
import os
from typing import Any, Sequence

from state_store.adapters import LinkedGroupStoreAdapter
from state_store.errors import CounterOverflowError
from state_store.sqlite import SqliteStateStore
from strategy_framework.linked_orders import (
    EmulatedLinkedOrderManager,
    GroupIdSpaceExhaustedError,
    GroupRecord,
    LinkedOrderError,
    group_record_from_dict,
    group_record_to_dict,
)

LOGGER = logging.getLogger(__name__)

SLOT_SIZE = 10**9
MAX_HOST_SLOT = 99


class SlotConflictError(LinkedOrderError):
    """An open linked leg in our host slot is unknown to our store: another host shares the slot."""


def group_id_of(record: GroupRecord) -> int:
    return record.group_id


def open_linked_group_store(
    path: "str | os.PathLike[str]",
) -> "tuple[SqliteStateStore, LinkedGroupStoreAdapter[GroupRecord]]":
    """Open (creating if needed) the state file and return it with a durable group store over it."""
    state_store = SqliteStateStore(path)
    group_store: LinkedGroupStoreAdapter[GroupRecord] = LinkedGroupStoreAdapter(
        state_store.records,
        to_dict=group_record_to_dict,
        from_dict=group_record_from_dict,
        group_id_of=group_id_of,
    )
    return state_store, group_store


def slot_bounds(host_slot: int) -> "tuple[int, int]":
    """Inclusive ``(first, last)`` group id of ``host_slot``."""
    return host_slot * SLOT_SIZE + 1, (host_slot + 1) * SLOT_SIZE - 1


def validate_host_slot(host_slot: int, max_group_id: int) -> None:
    if isinstance(host_slot, bool) or not isinstance(host_slot, int) or not 0 <= host_slot <= MAX_HOST_SLOT:
        raise ValueError(f"host_slot must be an int in 0..{MAX_HOST_SLOT}, got {host_slot!r}")
    if slot_bounds(host_slot)[1] > max_group_id:
        raise ValueError(
            f"host_slot {host_slot} needs group ids up to {slot_bounds(host_slot)[1]}, "
            f"above the codec max_group_id {max_group_id}"
        )


class FleetGroupIdAllocator:
    """``group_id = host_slot * 10**9 + seq`` with ``seq`` from a durable per-account, per-slot counter.

    The counter lives in the state file (outside the strategy DBs) and is reserved under ``BEGIN IMMEDIATE``,
    so it is unique across connections and processes and never reuses an id after a restart. The counter
    name holds only a hash of the account key.
    """

    def __init__(self, state_store: SqliteStateStore, *, account_key: str, host_slot: int, max_group_id: int):
        validate_host_slot(host_slot, max_group_id)
        self._counters = state_store.counters
        self._host_slot = host_slot
        self._floor, self._ceiling = slot_bounds(host_slot)
        digest = hashlib.sha256(account_key.encode("utf-8")).hexdigest()
        self._name = f"linked_group_id:{digest}:{host_slot}"

    @property
    def host_slot(self) -> int:
        return self._host_slot

    async def reserve_group_id(self) -> int:
        try:
            return await asyncio.to_thread(self._counters.reserve, self._name, self._floor, self._ceiling)
        except CounterOverflowError as exc:
            raise GroupIdSpaceExhaustedError(self._ceiling + 1, self._ceiling, "allocator") from exc


async def check_slot_conflict(
    venue: Any,
    codec: Any,
    store: Any,
    targets: Sequence[str],
    host_slot: int,
) -> None:
    """Raise ``SlotConflictError`` if an open linked leg lies in our slot but its group is not in our store."""
    first, last = slot_bounds(host_slot)
    known = {record.group_id for record in store.load_all()}
    for target in targets:
        for leg in await venue.list_open_legs(target):
            decoded = codec.decode(leg.client_order_id)
            if decoded is None:
                continue
            group_id = decoded[0]
            if first <= group_id <= last and group_id not in known:
                raise SlotConflictError(
                    f"open linked leg {leg.client_order_id} on {target} is in host slot {host_slot} but unknown to "
                    f"this host's store: another host is configured with the same slot. Refusing to place."
                )


async def recover_with_slot_check(
    manager: EmulatedLinkedOrderManager,
    venue: Any,
    codec: Any,
    store: Any,
    targets: Sequence[str],
    host_slot: int,
) -> None:
    """Slot-conflict check, then ``manager.recover``; a conflict leaves the manager un-recovered (no placing)."""
    try:
        await check_slot_conflict(venue, codec, store, targets, host_slot)
    except SlotConflictError:
        LOGGER.exception("Linked-order host slot conflict.")
        raise
    await manager.recover(list(targets))
