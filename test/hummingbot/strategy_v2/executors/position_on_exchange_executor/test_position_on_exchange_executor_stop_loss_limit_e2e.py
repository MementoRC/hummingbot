"""
End-to-end check that the STOP_LOSS_LIMIT ``limit_price`` produced by PositionOnExchangeExecutor survives the real
StrategyV2Base.buy/sell wrapper (``extra_kwargs``) and the Cython ``strategy_base`` kwargs merge, and arrives in the
connector's ``buy``/``sell`` call.

The connector here is a plain recording stub, so the test has no dependency on the market_connector sub-package. The
MarketConnectorExchange-level assertion (core receives trigger=TriggerSpec(ABSOLUTE, ...) and the limit price) belongs on
the branch that carries the adapter.
"""

from decimal import Decimal
from unittest.mock import MagicMock, PropertyMock, patch

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.core.time_iterator import TimeIterator
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.position_on_exchange_executor.data_types import (
    PositionOnExchangeExecutorConfig,
    PositionOnExchangeTripleBarrierConfig,
)
from hummingbot.strategy_v2.executors.position_on_exchange_executor.position_on_exchange_executor import (
    PositionOnExchangeExecutor,
)
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase

CONNECTOR_NAME = "binance"
TRADING_PAIR = "ETH-USDT"


class RecordingConnector(ConnectorBase):
    """Connector stub recording the kwargs of every buy/sell it receives.

    Built on ConnectorBase (whose c_buy/c_sell delegate to the Python buy/sell) rather than a paper exchange, whose
    Cython c_buy/c_sell bypass Python overrides.
    """

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, dict]] = []
        self.trading_rules = {TRADING_PAIR: TradingRule(trading_pair=TRADING_PAIR)}

    @property
    def ready(self) -> bool:
        return True

    def buy(self, trading_pair, amount, order_type, price, **kwargs) -> str:
        self.calls.append(("buy", {"order_type": order_type, "price": price, "amount": amount, **kwargs}))
        return "OID-BUY"

    def sell(self, trading_pair, amount, order_type, price, **kwargs) -> str:
        self.calls.append(("sell", {"order_type": order_type, "price": price, "amount": amount, **kwargs}))
        return "OID-SELL"


class TestStopLossLimitE2E(IsolatedAsyncioWrapperTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.connector = RecordingConnector()
        with patch("asyncio.create_task", return_value=MagicMock()):
            with patch(
                "hummingbot.strategy.strategy_v2_base.StrategyV2Base.listen_to_executor_actions",
                return_value=MagicMock(),
            ):
                with patch("hummingbot.strategy.strategy_v2_base.ExecutorOrchestrator"):
                    with patch("hummingbot.strategy.strategy_v2_base.MarketDataProvider"):
                        self.strategy = StrategyV2Base({CONNECTOR_NAME: self.connector})
        # Seed the strategy/order-tracker timestamp (NaN until first tick) without running StrategyV2Base.tick.
        TimeIterator.tick(self.strategy, 1234567890.0)

    def _build_executor(self, side: TradeType, order_type: OrderType, offset: Decimal | None):
        config = PositionOnExchangeExecutorConfig(
            id="e2e",
            timestamp=1234567890,
            trading_pair=TRADING_PAIR,
            connector_name=CONNECTOR_NAME,
            side=side,
            entry_price=Decimal("100"),
            amount=Decimal("1"),
            triple_barrier_config=PositionOnExchangeTripleBarrierConfig(
                stop_loss=Decimal("0.05"),
                take_profit=Decimal("0.1"),
                time_limit=60,
                take_profit_order_type=OrderType.LIMIT,
                stop_loss_order_type=order_type,
                stop_loss_limit_offset=offset,
            ),
        )
        return PositionOnExchangeExecutor(self.strategy, config)

    def _place_stop_loss(self, executor: PositionOnExchangeExecutor) -> tuple[str, dict]:
        with patch.object(
            PositionOnExchangeExecutor, "amount_to_close", new_callable=PropertyMock, return_value=Decimal("1")
        ):
            executor.place_stop_loss_order()
        self.assertEqual(1, len(self.connector.calls))
        return self.connector.calls[0]

    def test_long_stop_loss_limit_sell_carries_limit_price(self):
        executor = self._build_executor(TradeType.BUY, OrderType.STOP_LOSS_LIMIT, Decimal("0.002"))

        method, kwargs = self._place_stop_loss(executor)

        self.assertEqual("sell", method)
        self.assertEqual(OrderType.STOP_LOSS_LIMIT, kwargs["order_type"])
        self.assertEqual(Decimal("95"), kwargs["price"])
        self.assertEqual(Decimal("95") * (1 - Decimal("0.002")), kwargs["limit_price"])
        self.assertEqual(PositionAction.CLOSE, kwargs["position_action"])

    def test_short_stop_loss_limit_buy_carries_limit_price(self):
        executor = self._build_executor(TradeType.SELL, OrderType.STOP_LOSS_LIMIT, Decimal("0.002"))

        method, kwargs = self._place_stop_loss(executor)

        self.assertEqual("buy", method)
        self.assertEqual(OrderType.STOP_LOSS_LIMIT, kwargs["order_type"])
        self.assertEqual(Decimal("105"), kwargs["price"])
        self.assertEqual(Decimal("105") * (1 + Decimal("0.002")), kwargs["limit_price"])
        self.assertEqual(PositionAction.CLOSE, kwargs["position_action"])

    def test_plain_stop_loss_has_no_limit_price(self):
        for side, expected_method in ((TradeType.BUY, "sell"), (TradeType.SELL, "buy")):
            with self.subTest(side=side):
                self.connector.calls.clear()
                executor = self._build_executor(side, OrderType.STOP_LOSS, None)

                method, kwargs = self._place_stop_loss(executor)

                self.assertEqual(expected_method, method)
                self.assertEqual(OrderType.STOP_LOSS, kwargs["order_type"])
                self.assertNotIn("limit_price", kwargs)
