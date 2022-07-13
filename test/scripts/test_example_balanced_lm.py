import unittest
from decimal import Decimal
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, call, patch

import pandas as pd
from numpy import double
from sqlalchemy.util import asyncio

from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.connector.exchange.paper_trade.paper_trade_exchange import QuantizationParams
from hummingbot.connector.test_support.mock_paper_exchange import MockPaperExchange
from hummingbot.core.clock import Clock
from hummingbot.core.clock_mode import ClockMode
from hummingbot.core.data_type.common import TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.data_type.trade_fee import AddedToCostTradeFee
from hummingbot.core.event.event_logger import EventLogger
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    MarketEvent,
    OrderFilledEvent,
    OrderType,
    SellOrderCompletedEvent,
)
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from scripts.example_balanced_lm import ExampleBalancedLM, OrderStage
from scripts.funds_balancer import FundsBalancer


class TestExampleBalancedLM(unittest.TestCase):
    level = 0
    start: pd.Timestamp = pd.Timestamp("2019-01-01", tz="UTC")
    end: pd.Timestamp = pd.Timestamp("2019-01-01 01:00:00", tz="UTC")
    start_timestamp: float = start.timestamp()
    end_timestamp: float = end.timestamp()
    maker_trading_pairs: List[str] = ["COINALPHA-WETH", "COINALPHA", "WETH"]
    clock_tick_size = 10

    def handle(self, record):
        self.log_records.append(record)

    def _is_logged(self, log_level: str, message: str) -> bool:
        return any(record.levelname == log_level and record.getMessage().startswith(message)
                   for record in self.log_records)

    def setUp(self):
        self.clock: Clock = Clock(ClockMode.BACKTEST, self.clock_tick_size, self.start_timestamp, self.end_timestamp)
        self.ev_loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()

        self.log_records = []
        self.start: pd.Timestamp = pd.Timestamp("2019-01-01", tz="UTC")
        self.end: pd.Timestamp = pd.Timestamp("2019-01-01 01:00:00", tz="UTC")
        self.start_timestamp: float = self.start.timestamp()
        self.end_timestamp: float = self.end.timestamp()
        self.connector_name: str = "kucoin"
        self.clock_tick_size = 1
        self.connector: MockPaperExchange = MockPaperExchange(client_config_map=ClientConfigAdapter(ClientConfigMap()))

        self.markets: Dict[str, MockPaperExchange] = dict()
        self.markets_infos: Dict[str, Dict[str, MarketTradingPairTuple]] = dict()
        self.markets['kucoin_paper_trade'], self.markets_infos['kucoin_paper_trade'] = self.create_market(
            ['ALGO-USDT', 'AVAX-USDT', 'ADA-USDT', 'BTC-USDT', 'ETH-USDT', 'ALGO-AVAX', 'BTC-ALGO', 'BTC-ETH',
             'ETH-ADA'],
            [Decimal('0.5'), Decimal('0.25'), Decimal('1'), Decimal('20000'), Decimal('2000'),
             Decimal('0.5') / Decimal('0.25'), Decimal('20000') / Decimal('0.5'), Decimal('20000') / Decimal('2000'),
             Decimal('2000') / Decimal('1')], dict(ALGO=2000, AVAX=300, BTC=0.15, ETH=0.3, USDT=0.0))
        self.markets['binance_paper_trade'], self.markets_infos['binance_paper_trade'] = self.create_market(
            ['ALGO-USDT', 'AVAX-USDT', 'ADA-USDT', 'BTC-USDT', 'ETH-USDT', 'ALGO-AVAX', 'BTC-ALGO', 'BTC-ETH',
             'ETH-ADA'],
            [Decimal('0.5'), Decimal('0.25'), Decimal('1'), Decimal('20000'), Decimal('2000'),
             Decimal('0.5') / Decimal('0.25'), Decimal('20000') / Decimal('0.5'), Decimal('20000') / Decimal('2000'),
             Decimal('2000') / Decimal('1')], dict(ALGO=2000, AVAX=300, BTC=0.15, ETH=0.3, USDT=0.0))

        self.mocked_connector = MagicMock()
        self.mocked_connector.name = "kucoin"
        self.mocked_connector.get_all_balances.return_value = \
            dict(ALGO=Decimal("2000.8"), AVAX=Decimal("300.8"), BTC=Decimal("0.15"), ETH=Decimal("0.3"), )
        self.mocked_connector.available_balances = \
            dict(ALGO=Decimal("1000.8"), AVAX=Decimal("300.8"), BTC=Decimal("0.15"), ETH=Decimal("0.3"), )

        self.expected_balances = \
            [['kucoin', 'ALGO', 2000.8, 1000.8, 10.0, 20008.0],
             ['kucoin', 'AVAX', 300.8, 300.8, 10.0, 3008.0],
             ['kucoin', 'BTC', 0.15, 0.15, 30000.0, 4500.0],
             ['kucoin', 'ETH', 0.3, 0.3, 1900.0, 570.0],
             ['kucoin', 'USDT', 0.0, 0.0, 1.0, 0.0],
             ]
        for asset in self.expected_balances:
            self.connector.set_balance(asset[1], float(asset[2]))

        self.maker_order_fill_logger: EventLogger = EventLogger()
        self.cancel_order_logger: EventLogger = EventLogger()
        self.buy_order_completed_logger: EventLogger = EventLogger()
        self.sell_order_completed_logger: EventLogger = EventLogger()

        for c, v in self.markets.items():
            self.clock.add_iterator(v)
            v.add_listener(MarketEvent.BuyOrderCompleted, self.buy_order_completed_logger)
            v.add_listener(MarketEvent.SellOrderCompleted, self.sell_order_completed_logger)
            v.add_listener(MarketEvent.OrderFilled, self.maker_order_fill_logger)
            v.add_listener(MarketEvent.OrderCancelled, self.cancel_order_logger)

        ExampleBalancedLM._config = \
            dict(
                markets=dict(kucoin=dict(quotes=dict(USDT={'ALGO', 'AVAX'}, ETH={'ALGO'}, BTC={'AVAX'}),
                                         weights={'ALGO-USDT': 1.5},
                                         hbot=set(['ALGO-USDT']),
                                         inventory_skews={"ALGO-USDT": 30},
                                         hbot_weights={"ALGO-USDT": 1.5})),
                hbot_weight=1,
                inventory_skew=50,
                base_currency='USDT'
            )

        self.strategy = ExampleBalancedLM({self.connector_name: self.connector})
        self.strategy._prices = {'kucoin': {'ALGO-USDT': Decimal('0.5'),
                                            'AVAX-USDT': Decimal('0.25'),
                                            'ADA-USDT': Decimal('1'),
                                            'BTC-USDT': Decimal('20000'),
                                            'ETH-USDT': Decimal('2000'),
                                            'ALGO-AVAX': Decimal('0.5') / Decimal('0.25'),
                                            'BTC-ALGO': Decimal('20000') / Decimal('0.5'),
                                            'BTC-ETH': Decimal('20000') / Decimal('2000'),
                                            'ETH-ADA': Decimal('2000') / Decimal('1'),
                                            }}
        self.strategy.logger().setLevel(1)
        self.strategy.logger().addHandler(self)

        self.multi_strategy = ExampleBalancedLM(self.markets)
        self.multi_strategy.logger().setLevel(1)
        self.multi_strategy.logger().addHandler(self)
        self.multi_strategy.order_tracker._set_current_timestamp(1640001112.223)

    @staticmethod
    def create_market(trading_pairs: List[str], mid_price: List[Decimal], balances: Dict[str, int]) -> \
            (MockPaperExchange, Dict[str, MarketTradingPairTuple]):
        """
        Create a BacktestMarket and marketinfo dictionary to be used by the liquidity mining strategy
        """
        market: MockPaperExchange = MockPaperExchange(
            client_config_map=ClientConfigAdapter(ClientConfigMap())
        )
        market_infos: Dict[str, MarketTradingPairTuple] = {}

        for index, trading_pair in enumerate(trading_pairs):
            base_asset = trading_pair.split("-")[0]
            quote_asset = trading_pair.split("-")[1]
            market.set_balanced_order_book(trading_pair=trading_pair,
                                           mid_price=double(mid_price[index]),
                                           min_price=1,
                                           max_price=200,
                                           price_step_size=1,
                                           volume_step_size=10)
            market.set_quantization_param(QuantizationParams(trading_pair, 6, 6, 6, 6))
            market_infos[trading_pair] = MarketTradingPairTuple(market, trading_pair, base_asset, quote_asset)

        for asset, value in balances.items():
            market.set_balance(asset, value)

        return market, market_infos

    @staticmethod
    def simulate_limit_order_fill(market: MockPaperExchange, order_candidate: OrderCandidate, order_id: str = "0"):
        market.trigger_event(MarketEvent.OrderFilled, OrderFilledEvent(
            timestamp=market.current_timestamp,
            order_id=order_id,
            trading_pair=order_candidate.trading_pair,
            order_type=order_candidate.order_type,
            trade_type=order_candidate.order_side,
            price=order_candidate.price,
            amount=order_candidate.amount,
            trade_fee=AddedToCostTradeFee(Decimal("0"))
        ))
        if order_candidate.order_side == TradeType.BUY:
            market.trigger_event(MarketEvent.BuyOrderCompleted, BuyOrderCompletedEvent(
                timestamp=market.current_timestamp,
                order_id=order_id,
                base_asset=order_candidate.trading_pair.split('-')[0],
                quote_asset=order_candidate.trading_pair.split('-')[1],
                base_asset_amount=order_candidate.amount,
                quote_asset_amount=order_candidate.amount * order_candidate.price,
                order_type=order_candidate.order_type
            ))
        else:
            market.trigger_event(MarketEvent.SellOrderCompleted, SellOrderCompletedEvent(
                timestamp=market.current_timestamp,
                order_id=order_id,
                base_asset=order_candidate.trading_pair.split('-')[0],
                quote_asset=order_candidate.trading_pair.split('-')[1],
                base_asset_amount=order_candidate.amount,
                quote_asset_amount=order_candidate.amount * order_candidate.price,
                order_type=order_candidate.order_type
            ))

    @patch('hummingbot.user.user_balances.UserBalances')
    def test__call_asyncs(self, mocked_user_balances):
        mocked_user_balances._UserBalances__instance.update_exchange_balance = AsyncMock()
        with patch('hummingbot.core.rate_oracle.rate_oracle.RateOracle.get_kucoin_prices',
                   return_value=self.strategy._prices):
            self.strategy._call_asyncs()
        self.assertEqual(mocked_user_balances.call_args_list, [])
        print(self.strategy._list_futures_done)
        self.assertEqual(len(self.strategy._list_futures_done), 2)

    def test_start(self):
        self.assertFalse(self.strategy.ready_to_trade)
        self.strategy.start(Clock(ClockMode.BACKTEST), self.start_timestamp)
        self.strategy.tick(self.start_timestamp + 10)
        self.assertTrue(self.strategy.ready_to_trade)
        self.assertEqual(self.strategy._tick_time, self.clock.tick_time())

    def test__find_trades_routes_single_sell(self):
        with patch.object(ExampleBalancedLM, 'get_campaigns_from_config') as mocked_campaigns:
            mocked_campaigns.return_value = ExampleBalancedLM._config['markets']['kucoin']
            with patch.object(ExampleBalancedLM, 'get_assets_from_config') as mocked_assets:
                mocked_assets.return_value = {'ALGO', 'AVAX', 'BTC', 'USDT', 'ETH'}
                with patch.object(FundsBalancer, 'balancing_proposal') as mocked_proposal:
                    mocked_proposal.return_value = [{'amount': 1, 'asset': 'ALGO', 'to': 'USDT'}]
                    self.strategy._find_trades_routes(self.mocked_connector, slippage=0, fee=0)
                    self.assertEqual(self.strategy._balancing_trades['kucoin'][0]['trades']['proceed'],
                                     (Decimal('0.5'),))
                    self.strategy._find_trades_routes(self.mocked_connector, slippage=0, fee=0.01)
                    self.assertEqual(self.strategy._balancing_trades['kucoin'][0]['trades']['proceed'],
                                     (Decimal('0.4949999999999999998959165914'),))
                    self.strategy._find_trades_routes(self.mocked_connector, slippage=0.01, fee=0)
                    self.assertEqual(self.strategy._balancing_trades['kucoin'][0]['trades']['proceed'],
                                     (Decimal('0.4949999999999999998959165914'),))

    def test__find_trades_routes_single_buy(self):
        with patch.object(ExampleBalancedLM, 'get_campaigns_from_config') as mocked_campaigns:
            mocked_campaigns.return_value = ExampleBalancedLM._config['markets']['kucoin']
            with patch.object(ExampleBalancedLM, 'get_assets_from_config') as mocked_assets:
                mocked_assets.return_value = {'ALGO', 'AVAX', 'BTC', 'USDT', 'ETH'}
                with patch.object(FundsBalancer, 'balancing_proposal') as mocked_proposal:
                    mocked_proposal.return_value = [{'amount': 1, 'asset': 'AVAX', 'to': 'ALGO'}]
                    self.strategy._find_trades_routes(self.mocked_connector, slippage=0, fee=0)
        self.assertEqual(self.strategy._balancing_trades['kucoin'][0]['trades']['proceed'], (Decimal('0.5'),))

    def test__find_trades_routes_double_sell(self):
        with patch.object(ExampleBalancedLM, 'get_campaigns_from_config') as mocked_campaigns:
            mocked_campaigns.return_value = ExampleBalancedLM._config['markets']['kucoin']
            with patch.object(ExampleBalancedLM, 'get_assets_from_config') as mocked_assets:
                mocked_assets.return_value = {'ALGO', 'AVAX', 'BTC', 'USDT', 'ETH'}
                with patch.object(FundsBalancer, 'balancing_proposal') as mocked_proposal:
                    mocked_proposal.return_value = [{'amount': 1, 'asset': 'ADA', 'to': 'BTC'}]
                    self.strategy._find_trades_routes(self.mocked_connector, slippage=0, fee=0)
        self.assertEqual(self.strategy._balancing_trades['kucoin'][0]['trades']['proceed'],
                         (Decimal('1'), Decimal('0.00005')))

    def test__find_trades_routes_double_buy(self):
        with patch.object(ExampleBalancedLM, 'get_campaigns_from_config') as mocked_campaigns:
            mocked_campaigns.return_value = ExampleBalancedLM._config['markets']['kucoin']
            with patch.object(ExampleBalancedLM, 'get_assets_from_config') as mocked_assets:
                mocked_assets.return_value = {'ALGO', 'AVAX', 'BTC', 'USDT', 'ETH'}
                with patch.object(FundsBalancer, 'balancing_proposal') as mocked_proposal:
                    mocked_proposal.return_value = [{'amount': 1, 'asset': 'ALGO', 'to': 'ETH'}]
                    self.strategy._find_trades_routes(self.mocked_connector, slippage=0, fee=0)
        self.assertEqual(self.strategy._balancing_trades['kucoin'][0]['trades']['proceed'],
                         (Decimal('0.5'), Decimal('0.00025')))
        # d = self.strategy._balancing_trades['kucoin'][0]['trades']
        # c = dict(zip(d, v)) for v in zip(*d.values())]
        # print(self.strategy._balancing_trades)
        # print([dict(zip(d, v)) for v in zip(*d.values())])

    @patch('hummingbot.user.user_balances.UserBalances')
    @patch('hummingbot.core.rate_oracle.rate_oracle.RateOracle')
    def test__refresh_balances_prices_routes_async(self, mocked_prices, mocked_balances):
        mocked_balances._UserBalances__instance.update_exchange_balance = AsyncMock()
        mocked_prices.get_kucoin_prices = AsyncMock()

        mocked_prices.get_kucoin_prices.return_value = self.strategy._prices['kucoin']

        # Test call of async functions
        self.strategy._asyncs_called = False
        self.strategy._list_futures_done = dict()
        self.strategy._refresh_balances_prices_routes()
        self.assertEqual(len(self.strategy._list_futures_done), 2)

        # Test asyncs_called True but not found in _list_future
        self.strategy._asyncs_called = True
        self.strategy._list_futures_done = dict()
        with self.assertRaises(KeyError):
            self.strategy._refresh_balances_prices_routes()
        self.assertEqual(len(self.strategy._list_futures_done), 0)

    def test__refresh_balances_prices_routes_futures(self):
        # Test asyncs_called True, one of the 2 futures done
        self.strategy._asyncs_called = True
        self.strategy._balances_fut['kucoin'] = MagicMock()
        self.strategy._balances_fut['kucoin'].done.return_value = True
        self.strategy._prices_fut['kucoin'] = MagicMock()
        self.strategy._prices_fut['kucoin'].done.return_value = False
        self.strategy._list_futures_done = dict()
        self.strategy._list_futures_done[self.strategy._balances_fut['kucoin']] = False
        self.strategy._list_futures_done[self.strategy._prices_fut['kucoin']] = False

        # Empty the prices to verify its correct fill
        self.strategy._prices['kucoin'] = dict()
        self.strategy._refresh_balances_prices_routes()
        self.assertEqual(self.strategy._list_futures_done[self.strategy._balances_fut['kucoin']], True)
        self.assertEqual(self.strategy._list_futures_done[self.strategy._prices_fut['kucoin']], False)
        self.assertEqual(self.strategy._prices['kucoin'], {})
        self.assertEqual(self.strategy._data_ready, False)

        # Future gets done
        expected_result = {'ALGO-USDT': Decimal('0.5'),
                           'AVAX-USDT': Decimal('0.25'),
                           'ADA-USDT': Decimal('1'),
                           'BTC-USDT': Decimal('20000'),
                           'ETH-USDT': Decimal('2000'),
                           'ALGO-AVAX': Decimal('0.5') / Decimal('0.25'),
                           'BTC-ALGO': Decimal('20000') / Decimal('0.5'),
                           'BTC-ETH': Decimal('20000') / Decimal('2000'),
                           'ETH-ADA': Decimal('2000') / Decimal('1'),
                           }

        # Mimic the future done()
        self.strategy._prices_fut['kucoin'].done.return_value = True
        self.strategy._prices_fut['kucoin'].result.return_value = expected_result
        with patch('hummingbot.core.rate_oracle.rate_oracle.RateOracle.get_kucoin_prices',
                   return_value=self.strategy._prices_fut['kucoin'].result.return_value):
            # Mocking _find_trades_routes that has its own test plan
            with patch.object(ExampleBalancedLM, '_find_trades_routes') as mocked_routes:
                mocked_routes.return_value = True
                self.strategy._refresh_balances_prices_routes()
                self.assertEqual(mocked_routes.called, True)
        # Has the list of done futures reset
        self.assertEqual(self.strategy._list_futures_done, {})
        self.assertEqual(self.strategy._prices_fut['kucoin'], None)
        self.assertEqual(self.strategy._balances_fut['kucoin'], None)
        self.assertEqual(self.strategy._prices['kucoin'], expected_result)
        self.assertEqual(self.strategy._data_ready, True)

    def test__create_proposal(self):
        self.strategy._balancing_trades = {'kucoin': [
            {'amount': 1, 'asset': 'ALGO', 'to': 'ETH', 'route': ['ALGO', 'USDT', 'ETH'],
             'trades': {'rates': (Decimal('0.5'), Decimal('0.0005')), 'pairs': ('ALGO-USDT', 'ETH-USDT'),
                        'orders': ('buy_quote', 'sell_quote'), 'amounts': (Decimal('1'), Decimal('0.00025')),
                        'proceed': (Decimal('0.5'), Decimal('0.00025'))}}]}
        with patch.object(ExampleBalancedLM, '_update_proposal_prices') as mocked_price:
            mocked_price.return_value = [
                OrderCandidate("ALGO-USDT", False, OrderType.LIMIT, TradeType.BUY, Decimal('1'), Decimal('0.5')),
                OrderCandidate("ETH-USDT", False, OrderType.LIMIT, TradeType.BUY, Decimal('0.00025'),
                               Decimal('0.0005'))]
            with patch('hummingbot.connector.budget_checker.BudgetChecker.adjust_candidates') as mocked_budget:
                mocked_budget.return_value = [
                    OrderCandidate("ALGO-USDT", False, OrderType.LIMIT, TradeType.BUY, Decimal('1'), Decimal('0.5')),
                    OrderCandidate("ETH-USDT", False, OrderType.LIMIT, TradeType.BUY, Decimal('0.00025'),
                                   Decimal('0.0005'))]

                orders_l = self.strategy._create_market_proposal(adjust_budget=True)
        mocked_price.assert_called_with(mocked_price.return_value, self.connector)
        mocked_budget.assert_called_with(mocked_price.return_value, all_or_none=False)
        self.assertEqual(orders_l[0].trading_pair, 'ALGO-USDT')
        self.assertEqual(orders_l[0].amount, Decimal('1'))
        self.assertEqual(orders_l[0].price, Decimal('0.5'))
        self.assertEqual(orders_l[0].is_maker, False)
        self.assertEqual(orders_l[1].trading_pair, 'ETH-USDT')
        self.assertEqual(orders_l[1].amount, Decimal('0.00025'))
        self.assertEqual(orders_l[1].price, Decimal('0.0005'))
        self.assertEqual(orders_l[1].is_maker, False)

    def test__update_proposal_prices(self):
        lod = [OrderCandidate("T-T0", False, OrderType.LIMIT, TradeType.BUY, 0, 0),
               OrderCandidate("T-T1", False, OrderType.LIMIT, TradeType.BUY, 0, 0)]
        with patch('hummingbot.connector.connector_base.ConnectorBase') as mocked_price:
            mocked_price.get_price.return_value = 12345
            self.strategy._update_proposal_prices(lod, mocked_price)
        self.assertEqual(lod[0].amount, Decimal(12345))
        self.assertEqual(lod[1].amount, Decimal(12345))

    def test__place_order(self):
        lod = [OrderCandidate("T-T0", False, OrderType.LIMIT, TradeType.BUY, 0, 0),
               OrderCandidate("T-T1", False, OrderType.LIMIT, TradeType.SELL, 0, 0)]
        with patch.object(ExampleBalancedLM, 'buy') as mocked_buy:
            mocked_buy.return_value = "buy_id"
            buy = self.strategy._place_order('kucoin', lod[0])
        with patch.object(ExampleBalancedLM, 'sell') as mocked_sell:
            mocked_sell.return_value = "sell_id"
            sell = self.strategy._place_order('kucoin', lod[1])
        self.assertEqual(buy, "_amount_not_positive_")
        self.assertEqual(sell, "_amount_not_positive_")

        lod = [OrderCandidate("T-T0", False, OrderType.LIMIT, TradeType.BUY, 1, 0),
               OrderCandidate("T-T1", False, OrderType.LIMIT, TradeType.SELL, 1, 0)]
        with patch.object(ExampleBalancedLM, 'buy') as mocked_buy:
            mocked_buy.return_value = "buy_id"
            buy = self.strategy._place_order('kucoin', lod[0])
        with patch.object(ExampleBalancedLM, 'sell') as mocked_sell:
            mocked_sell.return_value = "sell_id"
            sell = self.strategy._place_order('kucoin', lod[1])
        mocked_buy.assert_called_with('kucoin', 'T-T0', 1, OrderType.LIMIT, 0)
        mocked_sell.assert_called_with('kucoin', 'T-T1', 1, OrderType.LIMIT, 0)
        self.assertEqual(buy, "buy_id")
        self.assertEqual(sell, "sell_id")

    def test__dequeue_proposal_exchange(self):
        lod = [OrderCandidate("T-T0", False, OrderType.LIMIT, TradeType.BUY, Decimal("10"), Decimal("10"))]
        self.strategy._trade_proposals = {'kucoin': lod.copy(), 'kraken': lod.copy()}

        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["buy_id_0", "buy_id_0"]
            self.strategy._dequeue_proposal_exchange('kucoin', lod.copy())
            self.assertEqual(1, mocked_order.call_count)
            mocked_order.assert_called_with('kucoin', lod[0])
            self.assertEqual({'kucoin': 'buy_id_0'}, self.strategy._order_id)
            self.assertEqual({'kucoin': lod[0]}, self.strategy._order_data)
            self.assertEqual({'kucoin': OrderStage.PLACED}, self.strategy._order_stage)

    def test__dequeue_proposal_all_exchanges(self):
        lod = [OrderCandidate("T-T0", False, OrderType.LIMIT, TradeType.BUY, Decimal("10"), Decimal("10"))]
        self.strategy._trade_proposals = {'kucoin': lod.copy(), 'kraken': lod.copy()}

        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["buy_id_0", "buy_id_0"]
            self.strategy._dequeue_proposal_all_exchanges()
            self.assertEqual(2, mocked_order.call_count)
            mocked_order.assert_called_with('kraken', lod[0])
            self.assertEqual({'kraken': 'buy_id_0', 'kucoin': 'buy_id_0'}, self.strategy._order_id)
            self.assertEqual({'kraken': lod[0], 'kucoin': lod[0]}, self.strategy._order_data)
            self.assertEqual({'kraken': OrderStage.PLACED, 'kucoin': OrderStage.PLACED}, self.strategy._order_stage)

    def test__dequeue_proposal_all_exchanges_amount_0(self):
        lod = [OrderCandidate("T-T0", False, OrderType.LIMIT, TradeType.BUY, Decimal("10"), Decimal("0")),
               OrderCandidate("ETH-USDT", False, OrderType.LIMIT, TradeType.SELL, Decimal("10"), Decimal("10"))]
        self.strategy._trade_proposals = {'kucoin': lod.copy(), 'kraken': lod.copy()}

        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["", "buy_id_kucoin_1", "buy_id_kraken_0"]
            self.strategy._dequeue_proposal_all_exchanges()
            self.assertEqual(3, mocked_order.call_count)
            mocked_order.assert_has_calls([call('kucoin', lod[0]), call('kucoin', lod[1]), call('kraken', lod[0])])
            self.assertEqual({'kraken': 'buy_id_kraken_0', 'kucoin': 'buy_id_kucoin_1'}, self.strategy._order_id)
            self.assertEqual({'kraken': lod[0], 'kucoin': lod[1]}, self.strategy._order_data)
            self.assertEqual({'kraken': OrderStage.PLACED, 'kucoin': OrderStage.PLACED}, self.strategy._order_stage)

    def test__dequeue_proposal_on_order_id(self):
        lod = [OrderCandidate("T-T0", False, OrderType.LIMIT, TradeType.BUY, Decimal("10"), Decimal("10")),
               OrderCandidate("ETH-USDT", False, OrderType.LIMIT, TradeType.SELL, Decimal("10"), Decimal("10"))]
        self.strategy._trade_proposals = {'kucoin': [lod.copy()[1]], 'kraken': [lod.copy()[1]]}
        self.strategy._order_id = {'kucoin': 'buy_kucoin_0', 'kraken': 'buy_kraken_0'}
        self.strategy._order_data = {'kucoin': lod[0], 'kraken': lod[0]}
        self.strategy._order_stage = {'kraken': OrderStage.PLACED, 'kucoin': OrderStage.PLACED}

        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["buy_kucoin_1", "buy_kraken1"]
            self.strategy._dequeue_proposal_on_order_id('buy_kucoin_0')
            self.assertEqual(1, mocked_order.call_count)
            mocked_order.assert_called_with('kucoin', lod[1])
            self.assertEqual({'kraken': 'buy_kraken_0', 'kucoin': 'buy_kucoin_1'}, self.strategy._order_id)
            self.assertEqual({'kraken': lod[0], 'kucoin': lod[1]}, self.strategy._order_data)
            self.assertEqual({'kraken': OrderStage.PLACED, 'kucoin': OrderStage.PLACED}, self.strategy._order_stage)

    def test__verify_buy_sel_fill_event(self):
        buy = [OrderCandidate("ETH-USDT", False, OrderType.LIMIT, TradeType.BUY, Decimal('1'), Decimal('1000'))]

        # Place the order
        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["buy_id_binance"]
            self.multi_strategy._dequeue_proposal_exchange('binance_paper_trade', buy.copy())
        fill_event = OrderFilledEvent(
            timestamp=0,
            order_id='buy_id_binance',
            trading_pair='ETH-USDT',
            trade_type=TradeType.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal('1000'),
            amount=Decimal('1'),
            trade_fee=AddedToCostTradeFee(percent=Decimal('0'), percent_token=None, flat_fees=[]), exchange_trade_id='',
            leverage=1,
            position='NIL')
        buy_event = BuyOrderCompletedEvent(
            timestamp=0,
            order_id='buy_id_binance',
            base_asset='ETH',
            quote_asset='USDT',
            base_asset_amount=Decimal('1'),
            quote_asset_amount=Decimal('1000'),
            order_type=OrderType.LIMIT,
            exchange_order_id=None)
        self.assertTrue(self.multi_strategy._verify_buy_sell_fill_event('binance_paper_trade', fill_event))
        self.assertTrue(self.multi_strategy._verify_buy_sell_fill_event('binance_paper_trade', buy_event))

        sell = [OrderCandidate("ETH-USDT", False, OrderType.LIMIT, TradeType.SELL, Decimal('1'), Decimal('1000'))]
        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["sell_id_binance"]
            self.multi_strategy._dequeue_proposal_exchange('binance_paper_trade', sell.copy())
        fill_event = OrderFilledEvent(
            timestamp=0,
            order_id='sell_id_binance',
            trading_pair='ETH-USDT',
            trade_type=TradeType.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal('1000'),
            amount=Decimal('1'),
            trade_fee=AddedToCostTradeFee(percent=Decimal('0'), percent_token=None, flat_fees=[]), exchange_trade_id='',
            leverage=1,
            position='NIL')
        sell_event = BuyOrderCompletedEvent(
            timestamp=0,
            order_id='sell_id_binance',
            base_asset='ETH',
            quote_asset='USDT',
            base_asset_amount=Decimal('1'),
            quote_asset_amount=Decimal('1000'),
            order_type=OrderType.LIMIT,
            exchange_order_id=None)
        self.assertTrue(self.multi_strategy._verify_buy_sell_fill_event('binance_paper_trade', fill_event))
        self.assertTrue(self.multi_strategy._verify_buy_sell_fill_event('binance_paper_trade', sell_event))

    def test__verify_buy_sell_fill_event_not_found(self):
        buy = [OrderCandidate("ETH-USDT", False, OrderType.LIMIT, TradeType.BUY, Decimal('1'), Decimal('1000'))]

        # Place the order
        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["buy_id_binance"]
            self.multi_strategy._dequeue_proposal_exchange('binance_paper_trade', buy.copy())
        fill_event = OrderFilledEvent(
            timestamp=0,
            order_id='another_buy_id_binance',
            trading_pair='ETH-USDT',
            trade_type=TradeType.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal('1000'),
            amount=Decimal('1'),
            trade_fee=AddedToCostTradeFee(percent=Decimal('0'), percent_token=None, flat_fees=[]), exchange_trade_id='',
            leverage=1,
            position='NIL')
        buy_event = BuyOrderCompletedEvent(
            timestamp=0,
            order_id='another_buy_id_binance',
            base_asset='ETH',
            quote_asset='USDT',
            base_asset_amount=Decimal('1'),
            quote_asset_amount=Decimal('1000'),
            order_type=OrderType.LIMIT,
            exchange_order_id=None)
        self.assertFalse(self.multi_strategy._verify_buy_sell_fill_event('binance_paper_trade', fill_event))
        self.assertFalse(self.multi_strategy._verify_buy_sell_fill_event('binance_paper_trade', buy_event))

        sell = [OrderCandidate("ETH-USDT", False, OrderType.LIMIT, TradeType.SELL, Decimal('1'), Decimal('1000'))]
        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["sell_id_binance"]
            self.multi_strategy._dequeue_proposal_exchange('binance_paper_trade', sell.copy())
        fill_event = OrderFilledEvent(
            timestamp=0,
            order_id='another_sell_id_binance',
            trading_pair='ETH-USDT',
            trade_type=TradeType.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal('1000'),
            amount=Decimal('1'),
            trade_fee=AddedToCostTradeFee(percent=Decimal('0'), percent_token=None, flat_fees=[]), exchange_trade_id='',
            leverage=1,
            position='NIL')
        sell_event = BuyOrderCompletedEvent(
            timestamp=0,
            order_id='another_sell_id_binance',
            base_asset='ETH',
            quote_asset='USDT',
            base_asset_amount=Decimal('1'),
            quote_asset_amount=Decimal('1000'),
            order_type=OrderType.LIMIT,
            exchange_order_id=None)
        self.assertFalse(self.multi_strategy._verify_buy_sell_fill_event('binance_paper_trade', fill_event))
        self.assertFalse(self.multi_strategy._verify_buy_sell_fill_event('binance_paper_trade', sell_event))

    def test__is_event_match_strategy(self):
        buy = [OrderCandidate("ETH-USDT", False, OrderType.LIMIT, TradeType.BUY, Decimal('1'), Decimal('1000'))]
        self.strategy._trade_proposals = {'kucoin': buy.copy()}
        fill_event = OrderFilledEvent(
            timestamp=self.start_timestamp,
            order_id="buy_id_0",
            trading_pair=buy[0].trading_pair,
            order_type=buy[0].order_type,
            trade_type=buy[0].order_side,
            price=buy[0].price,
            amount=buy[0].amount,
            trade_fee=AddedToCostTradeFee(Decimal("0"))
        )
        # Register the trade
        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["buy_id_0"]
            self.strategy._dequeue_proposal_exchange('kucoin', buy.copy())
        self.assertEqual((True, 'kucoin'), self.strategy._is_event_match_strategy(fill_event.order_id))
        self.assertEqual((False, None), self.strategy._is_event_match_strategy("_nonexistent_order_id_"))

        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["buy_id_0", "buy_id_0"]
            self.strategy._dequeue_proposal_exchange('kucoin', buy.copy())
            self.strategy._dequeue_proposal_exchange('kraken', buy.copy())
        self.assertRaises(ValueError, self.strategy._is_event_match_strategy, fill_event.order_id)

    def test_did_complete_buy_order_single(self):
        self.clock.add_iterator(self.strategy)
        order_time_1 = self.start_timestamp + self.clock_tick_size
        self.clock.backtest_til(order_time_1)

        lod = [OrderCandidate("ALGO-USDT", False, OrderType.LIMIT, TradeType.BUY, Decimal("10"), Decimal("10")),
               OrderCandidate("ETH-USDT", False, OrderType.LIMIT, TradeType.SELL, Decimal("10"), Decimal("10"))]
        self.multi_strategy._trade_proposals = {'binance_paper_trade': lod.copy()}

        # First call, preceding_order_id set to 0
        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["buy_id_0", "sell_id_1"]
            self.multi_strategy._dequeue_proposal_all_exchanges()
            self.assertEqual(self.multi_strategy._order_id, {'binance_paper_trade': 'buy_id_0'})
            # Trigger call to did_fill_order and did_complete_buy_order
            self.simulate_limit_order_fill(self.markets['binance_paper_trade'],
                                           OrderCandidate("BTC-ETH", False, OrderType.LIMIT, TradeType.BUY,
                                                          Decimal("10"),
                                                          Decimal("10")), "5")
            # Incorrect order filled, the second order is not placed
            self.assertEqual([lod[1]], self.multi_strategy._trade_proposals['binance_paper_trade'])

            self.simulate_limit_order_fill(self.markets['binance_paper_trade'], lod[0], "buy_id_0")
            # Correct order filled, the second order placed
            self.assertEqual([], self.multi_strategy._trade_proposals['binance_paper_trade'])
            self.assertEqual({'binance_paper_trade': 'sell_id_1'}, self.multi_strategy._order_id)

            # Order not placed but filled, does not dequeue
            self.simulate_limit_order_fill(self.markets['binance_paper_trade'], lod[1], "sell_id_1")
            self.assertEqual({'binance_paper_trade': 'sell_id_1'}, self.multi_strategy._order_id)

    def test_did_complete_buy_order_double(self):
        self.clock.add_iterator(self.strategy)
        order_time_1 = self.start_timestamp + self.clock_tick_size
        self.clock.backtest_til(order_time_1)

        lod = [OrderCandidate("ALGO-USDT", False, OrderType.LIMIT, TradeType.BUY, Decimal("10"), Decimal("10")),
               OrderCandidate("ETH-USDT", False, OrderType.LIMIT, TradeType.SELL, Decimal("10"), Decimal("10"))]
        print(OrderCandidate("ALGO-USDT", False, OrderType.LIMIT, TradeType.BUY, Decimal("10"), Decimal("10")))
        self.multi_strategy._trade_proposals = {'binance_paper_trade': lod.copy(), 'kucoin_paper_trade': lod.copy()}

        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["buy_id_binance", "buy_id_kucoin", "sell_id_binance", "sell_id_kucoin"]
            # First call, preceding_order_id set to 0
            self.multi_strategy._dequeue_proposal_all_exchanges()
            self.assertEqual({'binance_paper_trade': [lod[1]], 'kucoin_paper_trade': [lod[1]]},
                             self.multi_strategy._trade_proposals)
            self.assertEqual({'binance_paper_trade': 'buy_id_binance', 'kucoin_paper_trade': 'buy_id_kucoin'},
                             self.multi_strategy._order_id)

            self.simulate_limit_order_fill(self.markets['binance_paper_trade'],
                                           OrderCandidate("BTC-ETH", False, OrderType.LIMIT, TradeType.BUY,
                                                          Decimal("10"),
                                                          Decimal("10")), "buy_id_binance")
            self.assertEqual({'binance_paper_trade': [], 'kucoin_paper_trade': [lod[1]]},
                             self.multi_strategy._trade_proposals)

            self.simulate_limit_order_fill(self.markets['kucoin_paper_trade'],
                                           OrderCandidate("BTC-ETH", False, OrderType.LIMIT, TradeType.BUY,
                                                          Decimal("10"),
                                                          Decimal("10")), "buy_id_kucoin")
            self.assertEqual({'binance_paper_trade': 'sell_id_binance', 'kucoin_paper_trade': 'sell_id_kucoin'},
                             self.multi_strategy._order_id)
            self.assertEqual({'binance_paper_trade': [], 'kucoin_paper_trade': []},
                             self.multi_strategy._trade_proposals)
