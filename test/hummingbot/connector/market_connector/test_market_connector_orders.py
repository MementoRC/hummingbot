from decimal import Decimal
import unittest

import pytest

pytest.importorskip("market_connector")

from data_type_primitives import TradeFeeBase as PrimitiveFee, TradeFeeSchema  # noqa: E402
from market_connector.hb_compat import order_events as mc  # noqa: E402

from hummingbot.connector.market_connector.events_converter import EVENT_CLASSES  # noqa: E402
from hummingbot.connector.market_connector.market_connector_exchange import MarketConnectorExchange  # noqa: E402
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType  # noqa: E402
from hummingbot.core.data_type.trade_fee import TradeFeeBase  # noqa: E402
from hummingbot.core.event.event_forwarder import EventForwarder  # noqa: E402
from hummingbot.core.event.event_logger import EventLogger  # noqa: E402
from hummingbot.core.event.events import MarketEvent  # noqa: E402

from .fakes import PAIR, FakeDescriptor  # noqa: E402

D = Decimal


def _filled(cid="c1", trade_id="T1"):
    fee = PrimitiveFee.new_spot_fee(fee_schema=TradeFeeSchema(), trade_type=TradeType.BUY, percent=D("0.004"))
    return mc.OrderFilledEvent(
        1.0, cid, PAIR, TradeType.BUY, OrderType.LIMIT, D("100"), D("0.5"), fee, trade_id, "TX1", 1, "NIL"
    )


def _completed(cid="c1"):
    return mc.BuyOrderCompletedEvent(2.0, cid, "ETH", "USDC", D("1"), D("100"), OrderType.LIMIT, "TX1")


def _cancelled(cid="c1"):
    return mc.OrderCancelledEvent(2.0, cid, "TX1")


def _failure(cid="c1"):
    return mc.MarketOrderFailureEvent(2.0, cid, OrderType.LIMIT, "boom", "RuntimeError")


def _created(cid="c1"):
    return mc.BuyOrderCreatedEvent(1.0, OrderType.LIMIT, PAIR, D("1"), D("100"), cid, 1.0, "TX1")


def _sell_created(cid="c1"):
    return mc.SellOrderCreatedEvent(1.0, OrderType.LIMIT, PAIR, D("1"), D("100"), cid, 1.0, "TX1")


def _sell_completed(cid="c1"):
    return mc.SellOrderCompletedEvent(2.0, cid, "ETH", "USDC", D("1"), D("100"), OrderType.LIMIT, "TX1")


class OrderPathTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.descriptor = FakeDescriptor()
        self.core = self.descriptor.core
        self.exchange = MarketConnectorExchange(
            descriptor=self.descriptor,
            config={},
            connector_name="fake_mc",
            trading_pairs=[PAIR],
            order_book_refresh_interval=10,
        )
        self.loggers = {}
        for event in MarketEvent:
            if event.value in {eid for eid in EVENT_CLASSES}:
                logger = EventLogger()
                self.exchange.add_listener(event, logger)
                self.loggers[event.value] = logger
        await self.exchange.start_network()

    async def asyncTearDown(self):
        await self.exchange.stop_network()

    def _seen(self, event_id):
        return self.loggers[event_id].event_log

    # --- order entry -----------------------------------------------------------------------------------------

    async def test_buy_and_sell_delegate_and_return_cid(self):
        buy_cid = self.exchange.buy(PAIR, D("1"), OrderType.LIMIT, D("100"))
        sell_cid = self.exchange.sell(PAIR, D("2"), OrderType.MARKET)
        self.assertEqual("hbot-buy-1", buy_cid)
        self.assertEqual("hbot-sell-2", sell_cid)
        buy_call, sell_call = self.core.order_calls
        self.assertEqual(("buy", PAIR, D("1"), OrderType.LIMIT, D("100"), {}), buy_call)
        self.assertEqual(("sell", PAIR, D("2"), OrderType.MARKET), sell_call[:4])
        self.assertTrue(sell_call[4].is_nan())

    async def test_limit_type_forwards_limit_price(self):
        self.exchange.buy(PAIR, D("1"), OrderType.STOP_LOSS_LIMIT, D("95"), limit_price=D("94"), price_in_percent=False)
        extras = self.core.order_calls[0][5]
        self.assertEqual(D("94"), extras["limit_price"])
        self.assertIs(OrderType.STOP_LOSS_LIMIT, self.core.order_calls[0][3])

    async def test_position_action_and_expiration_pass_through_only_from_kwargs(self):
        self.exchange.sell(
            PAIR, D("1"), OrderType.LIMIT, D("100"), position_action=PositionAction.OPEN, expiration_ts=123.0, junk=1
        )
        extras = self.core.order_calls[0][5]
        self.assertEqual({"position_action": PositionAction.OPEN, "expiration_ts": 123.0}, extras)

    async def test_unsupported_type_raises_and_nothing_sent(self):
        with self.assertRaises(ValueError):
            self.exchange.buy(PAIR, D("1"), OrderType.LIMIT_MAKER, D("100"))
        self.assertEqual([], self.core.order_calls)

    async def test_order_entry_before_start_raises(self):
        idle = MarketConnectorExchange(self.descriptor, {}, "idle_mc", [PAIR])
        with self.assertRaises(RuntimeError):
            idle.buy(PAIR, D("1"), OrderType.LIMIT, D("100"))
        with self.assertRaises(RuntimeError):
            idle.cancel(PAIR, "c1")

    async def test_cancel_delegates_and_unknown_is_noop(self):
        self.assertEqual("ghost", self.exchange.cancel(PAIR, "ghost"))
        self.assertEqual([(PAIR, "ghost")], self.core.cancel_calls)
        self.assertEqual({}, self.exchange.in_flight_orders)

    # --- event translation -----------------------------------------------------------------------------------

    async def test_all_seven_ids_reach_trigger_event_with_parent_fields(self):
        events = {
            102: _completed(),
            103: _sell_completed(),
            106: _cancelled(),
            107: _filled(),
            198: _failure(),
            200: _created(),
            201: _sell_created(),
        }
        for event_id, event in events.items():
            with self.subTest(event_id=event_id):
                self.core.emit(event)
                (received,) = self._seen(event_id)
                self.assertIsInstance(received, EVENT_CLASSES[event_id][1])
                self.assertEqual("c1", received.order_id)
        filled = self._seen(107)[0]
        self.assertIsInstance(filled.trade_fee, TradeFeeBase)
        self.assertEqual(D("0.004"), filled.trade_fee.percent)

    async def test_stop_network_unsubscribes(self):
        bus = self.descriptor.create_calls[0][1]
        self.assertTrue(all(bus.get_subscribers(event_id) for event_id in EVENT_CLASSES))
        await self.exchange.stop_network()
        self.assertFalse(any(bus.get_subscribers(event_id) for event_id in EVENT_CLASSES))
        bus.publish(106, _cancelled())
        self.assertEqual([], self._seen(106))

    async def test_listener_exception_does_not_block_others(self):
        def explode(_):
            raise RuntimeError("listener boom")

        bad = EventForwarder(explode)
        self.exchange.add_listener(MarketEvent.OrderCancelled, bad)
        self.core.emit(_cancelled())
        self.core.emit(_filled())
        self.assertEqual(1, len(self._seen(106)))
        self.assertEqual(1, len(self._seen(107)))

    async def test_unknown_resolved_found_emits_created_once(self):
        self.core.emit(_created())
        self.assertEqual(1, len(self._seen(200)))
        self.assertEqual([], self._seen(198))

    async def test_unknown_resolved_missing_emits_failure_once(self):
        self.core.emit(_failure())
        self.assertEqual(1, len(self._seen(198)))
        self.assertEqual([], self._seen(200))

    async def test_cancel_race_fill_then_cancel_both_forwarded(self):
        self.core.emit(_filled())
        self.core.emit(_cancelled())
        self.assertEqual(1, len(self._seen(107)))
        self.assertEqual(1, len(self._seen(106)))

    async def test_late_fill_after_cancel_is_forwarded(self):
        self.core.emit(_cancelled())
        self.core.emit(_filled(trade_id="T-late"))
        self.assertEqual(["T-late"], [e.exchange_trade_id for e in self._seen(107)])
        self.assertEqual(1, len(self._seen(106)))

    async def test_late_fill_after_completion_not_dropped_not_duplicated(self):
        self.core.emit(_completed())
        self.core.emit(_filled(trade_id="T-late"))
        self.assertEqual(1, len(self._seen(102)))
        self.assertEqual(["T-late"], [e.exchange_trade_id for e in self._seen(107)])

    async def test_duplicate_terminal_events_suppressed(self):
        for _ in range(2):
            self.core.emit(_cancelled())
            self.core.emit(_completed())
            self.core.emit(_failure())
        self.assertEqual([1, 1, 1], [len(self._seen(i)) for i in (106, 102, 198)])
        self.core.emit(_cancelled("other"))
        self.assertEqual(2, len(self._seen(106)))
