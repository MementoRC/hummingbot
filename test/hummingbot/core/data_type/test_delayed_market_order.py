from decimal import Decimal
from unittest import TestCase

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.delayed_market_order import DelayedMarketOrder, DelayedOrderStatus


class DelayedMarketOrderTest(TestCase):
    def _make_order(self, status: DelayedOrderStatus = DelayedOrderStatus.PENDING) -> DelayedMarketOrder:
        order = DelayedMarketOrder(
            client_order_id="test-001",
            trading_pair="BTC-USDT",
            order_type=OrderType.STOP_LOSS,
            trade_type=TradeType.SELL,
            amount=Decimal("0.5"),
            trigger_price=Decimal("40000"),
        )
        order.status = status
        return order

    # --- Construction ---

    def test_create_delayed_order(self):
        order = self._make_order()
        self.assertEqual(order.client_order_id, "test-001")
        self.assertEqual(order.trading_pair, "BTC-USDT")
        self.assertEqual(order.order_type, OrderType.STOP_LOSS)
        self.assertEqual(order.trade_type, TradeType.SELL)
        self.assertEqual(order.amount, Decimal("0.5"))
        self.assertEqual(order.trigger_price, Decimal("40000"))

    def test_default_status_pending(self):
        order = DelayedMarketOrder(
            client_order_id="test-002",
            trading_pair="ETH-USDT",
            order_type=OrderType.TAKE_PROFIT,
            trade_type=TradeType.SELL,
            amount=Decimal("1.0"),
            trigger_price=Decimal("3000"),
        )
        self.assertEqual(order.status, DelayedOrderStatus.PENDING)

    def test_exchange_order_id_none_by_default(self):
        order = DelayedMarketOrder(
            client_order_id="test-003",
            trading_pair="BTC-USDT",
            order_type=OrderType.TRAILING_STOP,
            trade_type=TradeType.SELL,
            amount=Decimal("0.1"),
            trigger_price=Decimal("39000"),
        )
        self.assertIsNone(order.exchange_order_id)

    # --- is_pending ---

    def test_is_pending_true_when_pending(self):
        order = self._make_order(DelayedOrderStatus.PENDING)
        self.assertTrue(order.is_pending)

    def test_is_pending_false_when_triggered(self):
        order = self._make_order(DelayedOrderStatus.TRIGGERED)
        self.assertFalse(order.is_pending)

    def test_is_pending_false_when_filled(self):
        order = self._make_order(DelayedOrderStatus.FILLED)
        self.assertFalse(order.is_pending)

    # --- is_done ---

    def test_is_done_filled(self):
        order = self._make_order(DelayedOrderStatus.FILLED)
        self.assertTrue(order.is_done)

    def test_is_done_cancelled(self):
        order = self._make_order(DelayedOrderStatus.CANCELLED)
        self.assertTrue(order.is_done)

    def test_is_done_failed(self):
        order = self._make_order(DelayedOrderStatus.FAILED)
        self.assertTrue(order.is_done)

    def test_is_done_pending_false(self):
        order = self._make_order(DelayedOrderStatus.PENDING)
        self.assertFalse(order.is_done)

    def test_is_done_triggered_false(self):
        order = self._make_order(DelayedOrderStatus.TRIGGERED)
        self.assertFalse(order.is_done)
