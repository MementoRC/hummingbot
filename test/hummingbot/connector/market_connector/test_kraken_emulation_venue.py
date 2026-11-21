from decimal import Decimal
import unittest

import pytest

pytest.importorskip("market_connector")
pytest.importorskip("strategy_framework")

from data_type_primitives import (  # noqa: E402
    CancelOutcome,
    LegRole,
    LegSpec,
    LinkedOrderSpec,
    LinkKind,
    OrderOutcomeUnknownError,
    OrderState,
    TradeType,
)
from data_type_primitives.common import OrderType  # noqa: E402
from data_type_primitives.trigger_spec import TriggerKind, TriggerSpec  # noqa: E402
from market_connector.exchanges.kraken.linked_order_codec import KrakenClOrdIdLinkCodec  # noqa: E402
from market_connector.protocols import LegFill as McLegFill, LegSnapshot as McLegSnapshot  # noqa: E402
from strategy_framework.linked_orders import (  # noqa: E402
    EmulatedLinkedOrderManager,
    InMemoryLinkedGroupStore,
    LegFill,
    LegSnapshot,
    LinkedOrderManagerConfig,
)

from hummingbot.connector.market_connector.kraken_emulation_venue import KrakenEmulationVenue  # noqa: E402

from .fakes import PAIR, FakeGateway  # noqa: E402

D = Decimal
TAG = "mcl-1-0"
TRIGGER = TriggerSpec(kind=TriggerKind.ABSOLUTE, value=D("90"))


def _limit_leg(price="110", role=LegRole.OCO_LEG):
    return LegSpec(role=role, order_type=OrderType.LIMIT, trade_type=TradeType.SELL, amount=D("1"), price=D(price))


def _stop_leg():
    return LegSpec(
        role=LegRole.OCO_LEG, order_type=OrderType.STOP_LOSS, trade_type=TradeType.SELL, amount=D("1"), trigger=TRIGGER
    )


def _mc_snapshot(state, filled="0"):
    return McLegSnapshot(client_order_id=TAG, link_tag=TAG, state=state, filled=D(filled))


class KrakenEmulationVenueTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.gateway = FakeGateway()
        self.codec = KrakenClOrdIdLinkCodec()
        self.venue = KrakenEmulationVenue(self.gateway, self.codec)

    async def test_place_leg_returns_link_tag_not_txid(self):
        leg = _stop_leg()
        result = await self.venue.place_leg(PAIR, leg, link_tag=TAG)
        assert result == TAG
        assert result != self.gateway.place_return
        name, pair, order_type, side, amount, price, kwargs = self.gateway.calls[0]
        assert (name, pair, order_type, side, amount, price) == (
            "place_order",
            PAIR,
            OrderType.STOP_LOSS,
            TradeType.SELL,
            D("1"),
            None,
        )
        assert kwargs["cl_ord_id"] == TAG
        assert kwargs["trigger"] == TRIGGER

    async def test_place_leg_passes_limit_price(self):
        await self.venue.place_leg(PAIR, _limit_leg("110"), link_tag=TAG)
        assert self.gateway.calls[0][5] == D("110")

    async def test_place_leg_propagates_outcome_unknown(self):
        self.gateway.place_raises = OrderOutcomeUnknownError("lost reply", client_order_id=TAG)
        with pytest.raises(OrderOutcomeUnknownError):
            await self.venue.place_leg(PAIR, _limit_leg(), link_tag=TAG)

    async def test_cancel_true_canceled(self):
        outcome = await self.venue.cancel_leg(PAIR, TAG)
        assert outcome is CancelOutcome.CANCELED
        assert self.gateway.calls == [("cancel_order", PAIR, None, TAG)]

    async def test_cancel_false_filled_already_filled(self):
        self.gateway.cancel_return = False
        self.gateway.get_leg_return = _mc_snapshot(OrderState.FILLED, "1")
        assert await self.venue.cancel_leg(PAIR, TAG) is CancelOutcome.ALREADY_FILLED

    async def test_cancel_false_absent_or_canceled_not_found(self):
        self.gateway.cancel_return = False
        self.gateway.get_leg_return = None
        assert await self.venue.cancel_leg(PAIR, TAG) is CancelOutcome.NOT_FOUND
        self.gateway.get_leg_return = _mc_snapshot(OrderState.CANCELED)
        assert await self.venue.cancel_leg(PAIR, TAG) is CancelOutcome.NOT_FOUND

    async def test_cancel_false_lookup_error_unknown(self):
        self.gateway.cancel_return = False
        self.gateway.get_leg_raises = RuntimeError("down")
        assert await self.venue.cancel_leg(PAIR, TAG) is CancelOutcome.UNKNOWN

    async def test_get_leg_converts_snapshot(self):
        self.gateway.get_leg_return = McLegSnapshot(
            client_order_id=TAG,
            link_tag=TAG,
            state=OrderState.PARTIALLY_FILLED,
            filled=D("0.4"),
            venue_order_id="TX1",
            group_id="1",
        )
        snapshot = await self.venue.get_leg(PAIR, client_order_id=TAG)
        assert snapshot == LegSnapshot(
            client_order_id=TAG, link_tag=TAG, state=OrderState.PARTIALLY_FILLED, filled=D("0.4")
        )
        assert isinstance(snapshot, LegSnapshot)
        assert self.gateway.calls == [("get_leg", PAIR, TAG, None)]

    async def test_get_leg_none_passthrough(self):
        assert await self.venue.get_leg(PAIR, link_tag=TAG) is None

    async def test_list_open_legs_converts(self):
        self.gateway.open_legs = [_mc_snapshot(OrderState.OPEN)]
        legs = await self.venue.list_open_legs(PAIR)
        assert legs == [LegSnapshot(client_order_id=TAG, link_tag=TAG, state=OrderState.OPEN, filled=D("0"))]
        assert isinstance(legs[0], LegSnapshot)

    def test_supports_leg_by_capabilities(self):
        assert self.venue.supports_leg(PAIR, _limit_leg())
        assert self.venue.supports_leg(PAIR, _stop_leg())
        # Percent trailing-stop-limit is deferred (needs MC #298), even though Kraken lists the type.
        trailing_limit = LegSpec(
            role=LegRole.OCO_LEG,
            order_type=OrderType.TRAILING_STOP_LIMIT,
            trade_type=TradeType.SELL,
            amount=D("1"),
            trigger=TriggerSpec(kind=TriggerKind.PERCENT, value=D("0.01")),
            limit_offset=D("-1"),
        )
        assert not self.venue.supports_leg(PAIR, trailing_limit)

    async def test_subscribe_converts_fill_and_unsubscribes(self):
        received = []
        linked_id = self.codec.encode(1, 0)
        unsubscribe = self.venue.subscribe(received.append)
        await self.gateway.emit_fill(McLegFill(linked_id, D("0.5"), D("100"), 12.0, "T1"))
        assert received == [
            LegFill(
                client_order_id=linked_id,
                cumulative_filled=D("0.5"),
                last_price=D("100"),
                timestamp=12.0,
                fill_id="T1",
            )
        ]
        unsubscribe()
        assert self.gateway.unsubscribed == 1

    async def test_subscribe_drops_foreign_fill(self):
        received = []
        self.venue.subscribe(received.append)
        await self.gateway.emit_fill(McLegFill("hbot-buy-1", D("0.5"), D("100"), 12.0, "T1"))
        assert received == []

    async def test_subscribe_drops_fill_when_decode_raises(self):
        class _RaisingCodec:
            def decode(self, tag):
                raise ValueError("bad tag")

        venue = KrakenEmulationVenue(self.gateway, _RaisingCodec())
        received = []
        venue.subscribe(received.append)
        await self.gateway.emit_fill(McLegFill(self.codec.encode(1, 0), D("1"), D("100"), 1.0, "T1"))
        assert received == []

    async def test_fill_callback_reaches_on_fill_not_buffer(self):
        manager = EmulatedLinkedOrderManager(
            self.venue,
            KrakenClOrdIdLinkCodec(),
            LinkedOrderManagerConfig(enabled=True, require_recover_before_place=True, allow_volatile_store=True),
            store=InMemoryLinkedGroupStore(),
        )
        self.venue.subscribe(manager.on_fill)
        await manager.recover([PAIR])
        spec = LinkedOrderSpec(kind=LinkKind.OCO, legs=(_limit_leg(), _stop_leg()))
        handle = await manager.place_oco(PAIR, spec)
        tags = [call[6]["cl_ord_id"] for call in self.gateway.calls if call[0] == "place_order"]
        assert handle.leg_client_order_ids == tuple(tags)

        # A matched fill makes the manager cancel the sibling; a buffered (unmatched) one would not.
        await self.gateway.emit_fill(McLegFill(tags[0], D("1"), D("110"), 1.0, "T1"))
        cancels = [call for call in self.gateway.calls if call[0] == "cancel_order"]
        assert [call[3] for call in cancels] == [tags[1]]
