"""
End-to-end: the real PositionOnExchangeExecutor placing, tracking and cancelling exchange-native STOP_LOSS /
TAKE_PROFIT orders through the market-connector adapter, backed by the FakeCore / FakeDescriptor fakes.

The executor exists only in the modular tier; this module is skipped where it is absent.
"""

from decimal import Decimal
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

pytest.importorskip("market_connector")
pytest.importorskip("hummingbot.strategy_v2.executors.position_on_exchange_executor.position_on_exchange_executor")

from data_type_primitives import TradeFeeBase as PrimitiveFee, TradeFeeSchema  # noqa: E402
from data_type_primitives.trigger_spec import TriggerKind, TriggerSpec  # noqa: E402
from market_connector.hb_compat import order_events as mc  # noqa: E402

from hummingbot.connector.market_connector.events_converter import EVENT_CLASSES  # noqa: E402
from hummingbot.connector.market_connector.market_connector_exchange import MarketConnectorExchange  # noqa: E402
from hummingbot.core.data_type.common import OrderType, TradeType  # noqa: E402
from hummingbot.core.event.event_logger import EventLogger  # noqa: E402
from hummingbot.core.event.events import MarketEvent  # noqa: E402
from hummingbot.strategy.strategy_v2_base import StrategyV2Base  # noqa: E402
from hummingbot.strategy_v2.executors.position_on_exchange_executor.data_types import (  # noqa: E402
    PositionOnExchangeExecutorConfig,
    PositionOnExchangeTripleBarrierConfig,
)
from hummingbot.strategy_v2.executors.position_on_exchange_executor.position_on_exchange_executor import (  # noqa: E402
    PositionOnExchangeExecutor,
)
from hummingbot.strategy_v2.models.base import RunnableStatus  # noqa: E402
from hummingbot.strategy_v2.models.executors import CloseType  # noqa: E402
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase  # noqa: E402

from .fakes import PAIR, FakeDescriptor  # noqa: E402

D = Decimal
CONNECTOR = "kraken_mc"


def _sell_completed(cid):
    return mc.SellOrderCompletedEvent(2.0, cid, "ETH", "USDC", D("1"), D("95"), OrderType.STOP_LOSS, "TX1")


def _sell_filled(cid, trade_id):
    fee = PrimitiveFee.new_spot_fee(fee_schema=TradeFeeSchema(), trade_type=TradeType.SELL, percent=D("0.004"))
    return mc.OrderFilledEvent(
        1.0, cid, PAIR, TradeType.SELL, OrderType.STOP_LOSS, D("95"), D("1"), fee, trade_id, "TX1", 1, "NIL"
    )


def _cancelled(cid):
    return mc.OrderCancelledEvent(2.0, cid, "TX1")


class PositionOnExchangeExecutorKrakenMcTests(IsolatedAsyncioWrapperTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.descriptor = FakeDescriptor()
        self.core = self.descriptor.core
        self.exchange = MarketConnectorExchange(
            descriptor=self.descriptor,
            config={},
            connector_name=CONNECTOR,
            trading_pairs=[PAIR],
            order_book_refresh_interval=10,
        )
        self.loggers = {}
        for event in MarketEvent:
            if event.value in set(EVENT_CLASSES):
                logger = EventLogger()
                self.exchange.add_listener(event, logger)
                self.loggers[event.value] = logger
        await self.exchange.start_network()

        self.strategy = MagicMock(spec=StrategyV2Base)
        self.strategy.connectors = {CONNECTOR: self.exchange}
        self.strategy.buy.side_effect = lambda connector, pair, amount, order_type, price, position_action: (
            self.exchange.buy(pair, amount, order_type, price, position_action=position_action)
        )
        self.strategy.sell.side_effect = lambda connector, pair, amount, order_type, price, position_action: (
            self.exchange.sell(pair, amount, order_type, price, position_action=position_action)
        )
        self.strategy.cancel.side_effect = lambda connector_name, trading_pair, order_id: self.exchange.cancel(
            trading_pair, order_id
        )

        config = PositionOnExchangeExecutorConfig(
            id="e2e",
            timestamp=1234567890,
            trading_pair=PAIR,
            connector_name=CONNECTOR,
            side=TradeType.BUY,
            entry_price=D("100"),
            amount=D("1"),
            triple_barrier_config=PositionOnExchangeTripleBarrierConfig(
                stop_loss=D("0.05"),
                take_profit=D("0.1"),
                time_limit=60,
                take_profit_order_type=OrderType.TAKE_PROFIT,
                stop_loss_order_type=OrderType.STOP_LOSS,
            ),
        )
        self.executor = PositionOnExchangeExecutor(self.strategy, config)
        self.executor._status = RunnableStatus.RUNNING
        self.executor.register_events()
        patcher = patch.object(
            PositionOnExchangeExecutor, "amount_to_close", new_callable=PropertyMock, return_value=D("1")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def asyncTearDown(self):
        self.executor.unregister_events()
        await self.exchange.stop_network()
        await super().asyncTearDown()

    def _seen(self, event_id):
        return self.loggers[event_id].event_log

    def _place_stop_loss(self):
        self.executor.place_stop_loss_order()
        return self.executor._stop_loss_order.order_id

    # --- placement -------------------------------------------------------------------------------------------

    async def test_stop_loss_placed_with_absolute_trigger_and_close_action(self):
        order_id = self._place_stop_loss()
        side, pair, amount, order_type, price, extras = self.core.order_calls[0]
        self.assertEqual(("sell", PAIR, D("1")), (side, pair, amount))
        self.assertIs(OrderType.STOP_LOSS, order_type)
        self.assertTrue(price.is_nan())
        self.assertEqual(TriggerSpec(kind=TriggerKind.ABSOLUTE, value=self.executor.stop_loss_price), extras["trigger"])
        self.assertEqual("CLOSE", extras["position_action"].name)
        self.assertEqual("hbot-sell-1", order_id)

    async def test_take_profit_placed_with_absolute_trigger(self):
        self.executor.place_take_profit_order()
        _, _, _, order_type, price, extras = self.core.order_calls[0]
        self.assertIs(OrderType.TAKE_PROFIT, order_type)
        self.assertTrue(price.is_nan())
        self.assertEqual(
            TriggerSpec(kind=TriggerKind.ABSOLUTE, value=self.executor.take_profit_price), extras["trigger"]
        )
        self.assertEqual("CLOSE", extras["position_action"].name)

    # --- tracking --------------------------------------------------------------------------------------------

    async def test_stop_loss_completion_seen_once_despite_duplicate_terminal_event(self):
        order_id = self._place_stop_loss()
        self.core.emit(_sell_completed(order_id))
        self.core.emit(_sell_completed(order_id))
        self.assertEqual(1, len(self._seen(MarketEvent.SellOrderCompleted.value)))
        self.assertIs(CloseType.STOP_LOSS, self.executor.close_type)
        self.assertEqual(RunnableStatus.SHUTTING_DOWN, self.executor.status)

    async def test_late_fill_after_completion_forwarded_without_second_completion(self):
        order_id = self._place_stop_loss()
        self.core.emit(_sell_completed(order_id))
        self.core.emit(_sell_filled(order_id, "T-late"))
        self.assertEqual(1, len(self._seen(MarketEvent.SellOrderCompleted.value)))
        self.assertEqual(["T-late"], [e.exchange_trade_id for e in self._seen(MarketEvent.OrderFilled.value)])
        self.assertIs(CloseType.STOP_LOSS, self.executor.close_type)

    # --- cancel ----------------------------------------------------------------------------------------------

    async def test_cancel_stop_loss_routes_to_core_and_cancellation_handled_once(self):
        order_id = self._place_stop_loss()
        self.executor.cancel_stop_loss()
        self.assertEqual([(PAIR, order_id)], self.core.cancel_calls)
        self.core.emit(_cancelled(order_id))
        self.core.emit(_cancelled(order_id))
        self.assertEqual(1, len(self._seen(MarketEvent.OrderCancelled.value)))
        self.assertIsNone(self.executor._stop_loss_order)
        self.assertEqual(1, len(self.executor._failed_orders))

    async def test_cancel_take_profit_routes_to_core(self):
        self.executor.place_take_profit_order()
        order_id = self.executor._take_profit_order.order_id
        self.executor.cancel_take_profit()
        self.assertEqual([(PAIR, order_id)], self.core.cancel_calls)
