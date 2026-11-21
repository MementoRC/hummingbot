from decimal import Decimal
import unittest

import pytest

pytest.importorskip("market_connector")

from data_type_primitives import TradeFeeBase as PrimitiveFee, TradeFeeSchema as PrimitiveSchema  # noqa: E402
from market_connector.hb_compat import order_events as mc  # noqa: E402

from hummingbot.connector.market_connector import events_converter as conv  # noqa: E402
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType  # noqa: E402
from hummingbot.core.data_type.trade_fee import TradeFeeBase  # noqa: E402
from hummingbot.core.event import events as parent  # noqa: E402

D = Decimal


def _mc_events():
    fee = PrimitiveFee.new_spot_fee(fee_schema=PrimitiveSchema(), trade_type=TradeType.BUY, percent=D("0.004"))
    return {
        102: mc.BuyOrderCompletedEvent(1.0, "c1", "ETH", "USDC", D("1"), D("100"), OrderType.LIMIT, "TX1"),
        103: mc.SellOrderCompletedEvent(1.0, "c1", "ETH", "USDC", D("1"), D("100"), OrderType.MARKET, "TX1"),
        106: mc.OrderCancelledEvent(1.0, "c1", "TX1"),
        107: mc.OrderFilledEvent(
            1.0, "c1", "ETH-USDC", TradeType.BUY, OrderType.LIMIT, D("100"), D("0.5"), fee, "T9", "TX1", 1, "NIL"
        ),
        198: mc.MarketOrderFailureEvent(1.0, "c1", OrderType.LIMIT, "boom", "RuntimeError"),
        200: mc.BuyOrderCreatedEvent(1.0, OrderType.LIMIT, "ETH-USDC", D("1"), D("100"), "c1", 1.0, "TX1"),
        201: mc.SellOrderCreatedEvent(1.0, OrderType.STOP_LOSS_LIMIT, "ETH-USDC", D("1"), D("100"), "c1", 1.0, "TX1"),
    }


class EventsConverterTests(unittest.TestCase):
    def test_ids_match_core_registry(self):
        self.assertEqual(
            {name: event_id for event_id, (_, cls) in conv.EVENT_CLASSES.items() for name in [cls.__name__]},
            mc.MARKET_EVENT_IDS,
        )
        for event_id, event in conv.MARKET_EVENT_BY_ID.items():
            self.assertEqual(event_id, event.value)

    def test_every_id_converts_to_parent_class_with_fields(self):
        for event_id, mc_event in _mc_events().items():
            with self.subTest(event_id=event_id):
                result = conv.to_parent_event(event_id, mc_event)
                self.assertIsInstance(result, conv.EVENT_CLASSES[event_id][1])
                self.assertEqual("c1", result.order_id)
                self.assertEqual(1.0, result.timestamp)

    def test_filled_event_fields_and_parent_fee(self):
        result = conv.to_parent_event(107, _mc_events()[107])
        self.assertIsInstance(result, parent.OrderFilledEvent)
        self.assertIsInstance(result.trade_fee, TradeFeeBase)
        self.assertEqual(D("0.004"), result.trade_fee.percent)
        self.assertEqual(
            (D("100"), D("0.5"), "T9", "TX1"),
            (result.price, result.amount, result.exchange_trade_id, result.exchange_order_id),
        )
        self.assertIs(TradeType.BUY, result.trade_type)
        self.assertIs(OrderType.LIMIT, result.order_type)

    def test_created_and_failure_fields(self):
        created = conv.to_parent_event(201, _mc_events()[201])
        self.assertIs(OrderType.STOP_LOSS_LIMIT, created.type)
        self.assertEqual(D("1"), created.amount)
        failure = conv.to_parent_event(198, _mc_events()[198])
        self.assertEqual(("boom", "RuntimeError"), (failure.error_message, failure.error_type))

    def test_unknown_event_id_raises(self):
        with self.assertRaises(KeyError):
            conv.to_parent_event(999, object())

    def test_enum_converters_by_name(self):
        self.assertIs(OrderType.LIMIT, conv.to_parent_order_type(OrderType.LIMIT))
        self.assertIs(TradeType.SELL, conv.to_parent_trade_type(TradeType.SELL))
        self.assertIs(PositionAction.OPEN, conv.to_parent_position_action(PositionAction.OPEN))

    def test_fee_falls_back_to_spot_fee_when_not_convertible(self):
        class Odd:
            percent = D("0.01")

        fee = conv.to_parent_fee(Odd(), TradeType.BUY)
        self.assertIsInstance(fee, TradeFeeBase)
        self.assertEqual(D("0.01"), fee.percent)

    def test_terminal_deduper(self):
        deduper = conv.TerminalEventDeduper(max_entries=2)
        self.assertFalse(deduper.is_duplicate(106, "a"))
        self.assertTrue(deduper.is_duplicate(106, "a"))
        self.assertFalse(deduper.is_duplicate(102, "a"))
        self.assertFalse(deduper.is_duplicate(107, "a"))
        self.assertFalse(deduper.is_duplicate(107, "a"))
        self.assertFalse(deduper.is_duplicate(106, "b"))  # evicts the oldest
        self.assertFalse(deduper.is_duplicate(106, "a"))
        deduper.clear()
        self.assertFalse(deduper.is_duplicate(106, "b"))
