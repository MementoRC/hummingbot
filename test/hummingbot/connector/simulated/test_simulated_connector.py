import asyncio
from decimal import Decimal
import unittest

import pytest

pytest.importorskip("market_simulator.connector")

from market_simulator.connector import SimulatedConnectorCore  # noqa: E402
from market_simulator.simulator.exchange import SimulatedExchangeConfig  # noqa: E402
from market_simulator.simulator.fee_model import ZeroFeeModel  # noqa: E402
from market_simulator.simulator.matching_engine import ImmediateFillEngine  # noqa: E402

from hummingbot.connector.exchange_py_base import ExchangePyBase  # noqa: E402
from hummingbot.connector.simulated import SimulatedConnector  # noqa: E402
from hummingbot.core.data_type.common import OrderType, TradeType  # noqa: E402
from hummingbot.core.data_type.in_flight_order import OrderState  # noqa: E402
from hummingbot.core.event.event_logger import EventLogger  # noqa: E402
from hummingbot.core.event.events import MarketEvent  # noqa: E402

PAIR = "BTC-USDT"


def _make_core() -> SimulatedConnectorCore:
    config = SimulatedExchangeConfig(
        name="test_sim",
        trading_pairs=[PAIR],
        initial_balances={"USDT": Decimal("100000"), "BTC": Decimal("10")},
    )
    core = SimulatedConnectorCore(config=config, matching_engine=ImmediateFillEngine(), fee_model=ZeroFeeModel())
    core.sim_exchange.get_order_book(PAIR).apply_snapshot(
        bids=[(Decimal("50000"), Decimal("10"))],
        asks=[(Decimal("50100"), Decimal("10"))],
    )
    core.sim_exchange.ready = True
    return core


class SimulatedConnectorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.connector = SimulatedConnector(core=_make_core())
        await self.connector._update_trading_rules()
        await self.connector._update_balances()

    async def _settle(self):
        await asyncio.sleep(0.05)

    def _state(self, client_order_id: str) -> OrderState:
        return self.connector._order_tracker.fetch_order(client_order_id).current_state

    def test_is_exchange_py_base(self):
        self.assertIsInstance(self.connector, ExchangePyBase)
        self.assertEqual("test_sim", self.connector.name)

    def test_balances_and_trading_rules_populated(self):
        self.assertEqual(Decimal("100000"), self.connector.get_balance("USDT"))
        self.assertEqual(Decimal("10"), self.connector.get_available_balance("BTC"))
        self.assertIn(PAIR, self.connector.trading_rules)

    async def test_market_buy_fills_and_tracker_shows_filled(self):
        await self.connector._create_order(
            TradeType.BUY, "buy-1", PAIR, Decimal("0.1"), OrderType.MARKET, Decimal("50100")
        )
        await self._settle()

        order = self.connector._order_tracker.fetch_order("buy-1")
        self.assertEqual(OrderState.FILLED, order.current_state)
        self.assertEqual(Decimal("0.1"), order.executed_amount_base)
        self.assertIsNotNone(order.exchange_order_id)

    async def test_cancel_open_limit_gives_canceled_without_double_processing(self):
        cancelled = EventLogger()
        self.connector.add_listener(MarketEvent.OrderCancelled, cancelled)

        await self.connector._create_order(
            TradeType.BUY, "limit-1", PAIR, Decimal("0.1"), OrderType.LIMIT, Decimal("49000")
        )
        await self._settle()
        self.assertEqual(OrderState.OPEN, self._state("limit-1"))

        self.assertEqual("limit-1", await self.connector._execute_cancel(PAIR, "limit-1"))
        await self._settle()

        self.assertEqual(OrderState.CANCELED, self._state("limit-1"))
        self.assertEqual(1, len(cancelled.event_log))


if __name__ == "__main__":
    unittest.main()
