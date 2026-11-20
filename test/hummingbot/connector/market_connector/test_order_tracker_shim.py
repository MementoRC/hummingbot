from decimal import Decimal
import unittest

import pytest

pytest.importorskip("market_connector")

from data_type_primitives import (
    OrderState as CoreState,
    TradeFeeBase as PrimitiveFee,
    TradeFeeSchema,
    TradeType,
    TradeUpdate,
)  # noqa: E402,E501

from hummingbot.connector.market_connector.order_tracker_shim import CoreOrderTracker  # noqa: E402
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState  # noqa: E402
from hummingbot.core.data_type.trade_fee import TradeFeeBase  # noqa: E402

from .fakes import FakeCore, make_order  # noqa: E402


def _fill(order, trade_id="T1", base="0.5"):
    fee = PrimitiveFee.new_spot_fee(fee_schema=TradeFeeSchema(), trade_type=TradeType.BUY, percent=Decimal("0.004"))
    return TradeUpdate(
        trade_id=trade_id,
        client_order_id=order.client_order_id,
        exchange_order_id=order.exchange_order_id,
        trading_pair=order.trading_pair,
        fill_timestamp=1700000001.0,
        fill_price=Decimal("100"),
        fill_base_amount=Decimal(base),
        fill_quote_amount=Decimal(base) * 100,
        fee=fee,
    )


class CoreOrderTrackerTests(unittest.TestCase):
    def setUp(self):
        self.core = FakeCore()
        self.tracker = CoreOrderTracker(lambda: self.core)

    def test_conversion_preserves_decimals_and_fills(self):
        order = make_order(amount="1.25", price="100.10", state=CoreState.PARTIALLY_FILLED)
        order.update_with_trade_update(_fill(order, base="0.5"))
        self.core.add_order(order)
        converted = self.tracker.fetch_tracked_order("c1")
        self.assertIsInstance(converted, InFlightOrder)
        self.assertEqual((Decimal("1.25"), Decimal("100.10")), (converted.amount, converted.price))
        self.assertEqual(Decimal("0.5"), converted.executed_amount_base)
        self.assertEqual(Decimal("50.0"), converted.executed_amount_quote)
        self.assertIs(OrderState.PARTIALLY_FILLED, converted.current_state)
        self.assertEqual("TX1", converted.exchange_order_id)
        self.assertIsInstance(converted.order_fills["T1"].fee, TradeFeeBase)
        self.assertFalse(converted.is_done)
        self.assertEqual(["c1"], list(self.tracker.active_orders))
        self.assertEqual(self.tracker.active_orders, self.tracker.all_fillable_orders)
        self.assertEqual(self.tracker.active_orders, self.tracker.all_updatable_orders)

    def test_unknown_client_order_id_returns_none(self):
        self.assertIsNone(self.tracker.fetch_order(client_order_id="nope"))
        self.assertIsNone(self.tracker.fetch_order(exchange_order_id="nope"))
        self.assertIsNone(self.tracker.fetch_tracked_order("nope"))
        self.assertIsNone(self.tracker.fetch_order())

    def test_finished_order_fetchable_but_not_active(self):
        order = make_order(state=CoreState.FILLED)
        self.core.add_order(order, finished=True)
        fetched = self.tracker.fetch_order(client_order_id="c1")
        self.assertIs(OrderState.FILLED, fetched.current_state)
        self.assertTrue(fetched.is_done)
        self.assertEqual({}, self.tracker.active_orders)
        self.assertIsNone(self.tracker.fetch_tracked_order("c1"))
        self.assertEqual("c1", self.tracker.fetch_order(exchange_order_id="TX1").client_order_id)

    def test_core_unknown_state_maps_to_pending_create(self):
        self.core.add_order(make_order(state=CoreState.UNKNOWN, txid=None))
        converted = self.tracker.fetch_order(client_order_id="c1")
        self.assertIs(OrderState.PENDING_CREATE, converted.current_state)
        self.assertFalse(converted.is_done)

    def test_conversion_cached_until_order_changes(self):
        order = make_order(state=CoreState.OPEN)
        self.core.add_order(order)
        first = self.tracker.fetch_order(client_order_id="c1")
        self.assertIs(first, self.tracker.fetch_order(client_order_id="c1"))
        order.update_with_trade_update(_fill(order, base="1"))
        order.current_state = CoreState.FILLED
        second = self.tracker.fetch_order(client_order_id="c1")
        self.assertIsNot(first, second)
        self.assertIs(OrderState.FILLED, second.current_state)

    def test_no_core_yields_empty_view(self):
        tracker = CoreOrderTracker(lambda: None)
        self.assertEqual({}, tracker.active_orders)
        self.assertIsNone(tracker.fetch_order(client_order_id="c1"))
        self.assertIsNone(tracker.fetch_tracked_order("c1"))
