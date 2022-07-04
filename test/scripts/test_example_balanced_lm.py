import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
from sqlalchemy.util import asyncio

from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.connector.test_support.mock_paper_exchange import MockPaperExchange
from hummingbot.core.clock import Clock
from hummingbot.core.clock_mode import ClockMode
from hummingbot.core.data_type.common import TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderType
from scripts.example_balanced_lm import ExampleBalancedLM
from scripts.funds_balancer import FundsBalancer


class TestExampleBalancedLM(unittest.TestCase):
    level = 0

    def handle(self, record):
        self.log_records.append(record)

    def _is_logged(self, log_level: str, message: str) -> bool:
        return any(record.levelname == log_level and record.getMessage().startswith(message)
                   for record in self.log_records)

    def setUp(self):
        self.ev_loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()

        self.log_records = []
        self.start: pd.Timestamp = pd.Timestamp("2019-01-01", tz="UTC")
        self.end: pd.Timestamp = pd.Timestamp("2019-01-01 01:00:00", tz="UTC")
        self.start_timestamp: float = self.start.timestamp()
        self.end_timestamp: float = self.end.timestamp()
        self.connector_name: str = "kucoin"
        self.clock_tick_size = 1
        self.clock: Clock = Clock(ClockMode.BACKTEST, self.clock_tick_size, self.start_timestamp, self.end_timestamp)
        self.connector: MockPaperExchange = MockPaperExchange(client_config_map=ClientConfigAdapter(ClientConfigMap()))

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
        self.clock.add_iterator(self.connector)
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

                orders_l = self.strategy._create_proposal(adjust_budget=True)
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

    def test__dequeue_execute_proposal(self):
        lod = [OrderCandidate("T-T0", False, OrderType.LIMIT, TradeType.BUY, 1, 0),
               OrderCandidate("T-T1", False, OrderType.LIMIT, TradeType.SELL, 1, 0)]
        self.strategy._trade_proposals = {'kucoin': lod.copy(), 'kraken': lod.copy()}

        # First call, preceding_order_id set to 0
        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.side_effect = ["buy_id_0", "buy_id_1"]
            self.strategy._dequeue_execute_proposal("0")
        mocked_order.assert_called_with('kraken', lod[0])
        self.assertEqual(self.strategy._order_ids, {'kraken': 'buy_id_1', 'kucoin': 'buy_id_0'})

        # Second call, kraken first order is completed, placing second kraken order
        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.return_value = "buy_id_2"
            self.strategy._dequeue_execute_proposal("buy_id_1")
        mocked_order.assert_called_with('kraken', lod[1])
        self.assertEqual(self.strategy._order_ids, {'kraken': 'buy_id_2', 'kucoin': 'buy_id_0'})

        # Third call, kucoin first order is completed, placing second kucoin order
        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.return_value = "buy_id_3"
            self.strategy._dequeue_execute_proposal("buy_id_0")
        mocked_order.assert_called_with('kucoin', lod[1])
        self.assertEqual(self.strategy._order_ids, {'kraken': 'buy_id_2', 'kucoin': 'buy_id_3'})

        # Fourth call, kucoin second order is completed, we should have emptied the order queue
        with patch.object(ExampleBalancedLM, '_place_order') as mocked_order:
            mocked_order.return_value = "_not_orders_in_queue_"
            self.strategy._dequeue_execute_proposal("buy_id_3")
        assert not mocked_order.called, 'method should not have been called'
        self.assertEqual(self.strategy._order_ids, {'kraken': 'buy_id_2', 'kucoin': 'buy_id_3'})
