from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from hummingbot.connector.exchange_py_base import ExchangePyBase
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.event.events import MarketOrderFailureEvent, OrderCancelledEvent
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.position_on_exchange_executor.data_types import (
    PositionOnExchangeExecutorConfig,
    PositionOnExchangeTripleBarrierConfig,
)
from hummingbot.strategy_v2.executors.position_on_exchange_executor.position_on_exchange_executor import (
    PositionOnExchangeExecutor,
)
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import TrackedOrder
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase

MODULE = "hummingbot.strategy_v2.executors.position_on_exchange_executor.position_on_exchange_executor"


class TestPositionOnExchangeExecutorBranches(IsolatedAsyncioWrapperTestCase):
    def setUp(self) -> None:
        super().setUp()
        strategy = MagicMock(spec=StrategyV2Base)
        type(strategy).current_timestamp = PropertyMock(return_value=1234567890)
        strategy.cancel.return_value = None
        connector = MagicMock(spec=ExchangePyBase)
        strategy.connectors = {"binance": connector}
        self.strategy = strategy

    def _config(
        self,
        side: TradeType = TradeType.BUY,
        tp_type: OrderType = OrderType.TAKE_PROFIT,
        sl_type: OrderType = OrderType.STOP_LOSS,
    ) -> PositionOnExchangeExecutorConfig:
        barrier = PositionOnExchangeTripleBarrierConfig(
            stop_loss=Decimal("0.05"),
            take_profit=Decimal("0.1"),
            time_limit=60,
            take_profit_order_type=tp_type,
            stop_loss_order_type=sl_type,
        )
        return PositionOnExchangeExecutorConfig(
            id="test",
            timestamp=1234567890,
            trading_pair="ETH-USDT",
            connector_name="binance",
            side=side,
            entry_price=Decimal("100"),
            amount=Decimal("1"),
            triple_barrier_config=barrier,
        )

    def _executor(self, **kwargs) -> PositionOnExchangeExecutor:
        executor = PositionOnExchangeExecutor(self.strategy, self._config(**kwargs))
        executor._status = RunnableStatus.RUNNING
        return executor

    def test_init_restores_exchange_native_stop_loss_type(self):
        config = self._config(sl_type=OrderType.STOP_LOSS_LIMIT)
        PositionOnExchangeExecutor(self.strategy, config)
        self.assertEqual(config.triple_barrier_config.stop_loss_order_type, OrderType.STOP_LOSS_LIMIT)

    def test_stop_loss_price_by_side(self):
        with patch.object(PositionOnExchangeExecutor, "entry_price", new_callable=PropertyMock) as entry:
            entry.return_value = Decimal("100")
            self.assertEqual(self._executor(side=TradeType.BUY).stop_loss_price, Decimal("95.00"))
            self.assertEqual(self._executor(side=TradeType.SELL).stop_loss_price, Decimal("105.00"))

    def test_trade_pnl_pct_is_zero_while_close_order_in_flight(self):
        executor = self._executor()
        executor._close_order = TrackedOrder(order_id="OID-CLOSE")
        self.assertEqual(executor.trade_pnl_pct, Decimal("0"))

    def test_control_stop_loss_places_only_once(self):
        executor = self._executor()
        with patch.object(PositionOnExchangeExecutor, "place_order", return_value="OID-SL") as place_order:
            executor.control_stop_loss()
            executor.control_stop_loss()
        place_order.assert_called_once()
        self.assertEqual(executor._stop_loss_order.order_id, "OID-SL")

    def test_control_stop_loss_skipped_without_stop_loss_config(self):
        executor = self._executor()
        executor.config.triple_barrier_config.stop_loss = None
        with patch.object(PositionOnExchangeExecutor, "place_order") as place_order:
            executor.control_stop_loss()
        place_order.assert_not_called()

    def test_place_stop_loss_order_when_already_placed_is_noop(self):
        executor = self._executor()
        executor._stop_loss_order = TrackedOrder(order_id="OID-SL")
        with patch.object(PositionOnExchangeExecutor, "place_order") as place_order:
            executor.place_stop_loss_order()
        place_order.assert_not_called()
        self.assertEqual(executor._stop_loss_order.order_id, "OID-SL")

    def test_place_take_profit_order_uses_take_profit_type(self):
        executor = self._executor()
        with patch.object(PositionOnExchangeExecutor, "place_order", return_value="OID-TP") as place_order:
            executor.place_take_profit_order()
        self.assertEqual(place_order.call_args.kwargs["order_type"], OrderType.TAKE_PROFIT)
        self.assertEqual(executor._take_profit_order.order_id, "OID-TP")

    def test_place_take_profit_order_when_already_placed_is_noop(self):
        executor = self._executor()
        executor._take_profit_order = TrackedOrder(order_id="OID-TP")
        with patch.object(PositionOnExchangeExecutor, "place_order") as place_order:
            executor.place_take_profit_order()
        place_order.assert_not_called()

    def test_place_take_profit_order_with_limit_order_in_place_raises(self):
        executor = self._executor()
        executor._take_profit_limit_order = TrackedOrder(order_id="OID-TPL")
        with patch.object(PositionOnExchangeExecutor, "place_order") as place_order:
            with self.assertRaises(ValueError):
                executor.place_take_profit_order()
        place_order.assert_not_called()

    def test_renew_stop_loss_order_cancels_and_replaces(self):
        executor = self._executor()
        executor._stop_loss_order = TrackedOrder(order_id="OID-SL-OLD")
        with patch.object(PositionOnExchangeExecutor, "place_order", return_value="OID-SL-NEW"):
            executor.renew_stop_loss_order()
        self.strategy.cancel.assert_called_once_with(
            connector_name="binance", trading_pair="ETH-USDT", order_id="OID-SL-OLD"
        )
        self.assertEqual(executor._stop_loss_order.order_id, "OID-SL-NEW")

    def test_renew_take_profit_order_replaces_native_order(self):
        executor = self._executor()
        executor._take_profit_order = TrackedOrder(order_id="OID-TP-OLD")
        with patch.object(PositionOnExchangeExecutor, "place_order", return_value="OID-TP-NEW"):
            executor.renew_take_profit_order()
        self.strategy.cancel.assert_called_once_with(
            connector_name="binance", trading_pair="ETH-USDT", order_id="OID-TP-OLD"
        )
        self.assertEqual(executor._take_profit_order.order_id, "OID-TP-NEW")

    def test_renew_take_profit_order_replaces_limit_order(self):
        executor = self._executor(tp_type=OrderType.LIMIT)
        executor._take_profit_order = TrackedOrder(order_id="OID-TP-OLD")
        executor._take_profit_limit_order = TrackedOrder(order_id="OID-TPL")
        with (
            patch.object(PositionOnExchangeExecutor, "place_take_profit_limit_order") as place_limit,
            patch.object(PositionOnExchangeExecutor, "place_take_profit_order") as place_native,
        ):
            executor.renew_take_profit_order()
        place_limit.assert_called_once()
        place_native.assert_not_called()
        self.assertIsNone(executor._take_profit_order)

    def test_cancel_open_order(self):
        executor = self._executor()
        executor._open_order = TrackedOrder(order_id="OID-OPEN")
        executor.cancel_open_order()
        self.strategy.cancel.assert_called_once_with(
            connector_name="binance", trading_pair="ETH-USDT", order_id="OID-OPEN"
        )

    def test_cancel_open_orders_cancels_open_exchange_native_orders(self):
        executor = self._executor()
        executor._stop_loss_order = TrackedOrder(order_id="OID-SL")
        executor._stop_loss_order.order = MagicMock(is_open=True)
        executor._take_profit_order = TrackedOrder(order_id="OID-TP")
        executor._take_profit_order.order = MagicMock(is_open=True)
        executor.cancel_open_orders()
        cancelled = {call.kwargs["order_id"] for call in self.strategy.cancel.call_args_list}
        self.assertEqual(cancelled, {"OID-SL", "OID-TP"})

    def test_cancel_open_orders_skips_closed_exchange_native_orders(self):
        executor = self._executor()
        executor._stop_loss_order = TrackedOrder(order_id="OID-SL")
        executor._stop_loss_order.order = MagicMock(is_open=False)
        executor._take_profit_order = TrackedOrder(order_id="OID-TP")
        executor._take_profit_order.order = MagicMock(is_open=False)
        executor.cancel_open_orders()
        self.strategy.cancel.assert_not_called()

    def test_update_tracked_orders_with_order_id_updates_sl_and_tp(self):
        executor = self._executor()
        executor._stop_loss_order = TrackedOrder(order_id="OID-SL")
        executor._take_profit_order = TrackedOrder(order_id="OID-TP")
        sl_in_flight = MagicMock(name="sl_in_flight")
        tp_in_flight = MagicMock(name="tp_in_flight")
        lookup = {"OID-SL": sl_in_flight, "OID-TP": tp_in_flight}
        with patch.object(PositionOnExchangeExecutor, "get_in_flight_order", side_effect=lambda _c, oid: lookup[oid]):
            executor.update_tracked_orders_with_order_id("OID-SL")
            self.assertIs(executor._stop_loss_order.order, sl_in_flight)
            self.assertIsNone(executor._take_profit_order.order)
            executor.update_tracked_orders_with_order_id("OID-TP")
        self.assertIs(executor._take_profit_order.order, tp_in_flight)

    def test_process_order_canceled_event_for_stop_loss_and_take_profit(self):
        executor = self._executor()
        sl_order = TrackedOrder(order_id="OID-SL")
        tp_order = TrackedOrder(order_id="OID-TP")
        executor._stop_loss_order = sl_order
        executor._take_profit_order = tp_order
        executor.process_order_canceled_event("102", MagicMock(), OrderCancelledEvent(1234567890, "OID-SL"))
        self.assertIsNone(executor._stop_loss_order)
        self.assertIs(executor._take_profit_order, tp_order)
        executor.process_order_canceled_event("102", MagicMock(), OrderCancelledEvent(1234567890, "OID-TP"))
        self.assertIsNone(executor._take_profit_order)
        self.assertIn(sl_order, executor._failed_orders)
        self.assertIn(tp_order, executor._failed_orders)

    def test_process_order_failed_event_ignores_unrelated_orders(self):
        executor = self._executor()
        executor._stop_loss_order = TrackedOrder(order_id="OID-SL")
        executor._take_profit_order = TrackedOrder(order_id="OID-TP")
        executor.process_order_failed_event(
            "102",
            MagicMock(),
            MarketOrderFailureEvent(timestamp=1234567890, order_id="OID-OTHER", order_type=OrderType.MARKET),
        )
        self.assertIsNotNone(executor._stop_loss_order)
        self.assertIsNotNone(executor._take_profit_order)

    def test_control_take_profit_places_limit_order_within_activation_bounds(self):
        executor = self._executor(tp_type=OrderType.LIMIT)
        with (
            patch.object(PositionOnExchangeExecutor, "_is_within_activation_bounds", return_value=True),
            patch.object(PositionOnExchangeExecutor, "place_take_profit_limit_order") as place_limit,
        ):
            executor.control_take_profit()
        place_limit.assert_called_once()

    def test_control_take_profit_does_not_place_limit_order_outside_activation_bounds(self):
        executor = self._executor(tp_type=OrderType.LIMIT)
        with (
            patch.object(PositionOnExchangeExecutor, "_is_within_activation_bounds", return_value=False),
            patch.object(PositionOnExchangeExecutor, "place_take_profit_limit_order") as place_limit,
        ):
            executor.control_take_profit()
        place_limit.assert_not_called()

    def test_control_take_profit_cancels_open_limit_order_outside_activation_bounds(self):
        executor = self._executor(tp_type=OrderType.LIMIT)
        executor._take_profit_limit_order = MagicMock(is_open=True, is_filled=False)
        with (
            patch.object(PositionOnExchangeExecutor, "_is_within_activation_bounds", return_value=False),
            patch.object(PositionOnExchangeExecutor, "cancel_take_profit") as cancel_tp,
        ):
            executor.control_take_profit()
        cancel_tp.assert_called_once()

    def test_control_take_profit_keeps_limit_order_within_activation_bounds(self):
        executor = self._executor(tp_type=OrderType.LIMIT)
        executor._take_profit_limit_order = MagicMock(is_open=True, is_filled=False)
        with (
            patch.object(PositionOnExchangeExecutor, "_is_within_activation_bounds", return_value=True),
            patch.object(PositionOnExchangeExecutor, "cancel_take_profit") as cancel_tp,
        ):
            executor.control_take_profit()
        cancel_tp.assert_not_called()

    def test_control_take_profit_is_noop_when_native_order_exists(self):
        executor = self._executor()
        executor._take_profit_order = TrackedOrder(order_id="OID-TP")
        with patch.object(PositionOnExchangeExecutor, "place_take_profit_order") as place_native:
            executor.control_take_profit()
        place_native.assert_not_called()

    async def test_sleep_delegates_to_asyncio_sleep(self):
        executor = self._executor()
        with patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await executor._sleep(2.5)
        sleep.assert_awaited_once_with(2.5)
