import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import ruamel.yaml

from scripts.ls_fund_rebalancer import LiteStrategyFundRebalancer

yaml_parser = ruamel.yaml.YAML()


class TestLiteStrategyFundRebalancer(unittest.TestCase):
    def setUp(self) -> None:
        self.market_init = MagicMock()
        self.market_init.base_currency = "USDT"
        self.market_init.hbot_weight = 1
        self.market_init.inventory_skew = 50
        self.market_init._config = \
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
        self.market_init.get_assets_from_config.return_value = {'ALGO', 'AVAX', 'BTC', 'USDT', 'ETH'}
        self.market_init.get_campaigns_from_config.return_value = self.market_init._config['markets']['kucoin']

        self.mocked_connector = MagicMock()
        self.mocked_connector.get_all_balances.return_value = \
            dict(ALGO=Decimal("2000.8"), AVAX=Decimal("300.8"), BTC=Decimal("0.15"), ETH=Decimal("0.3"), )
        self.mocked_connector.available_balances = \
            dict(ALGO=Decimal("1000.8"), AVAX=Decimal("300.8"), BTC=Decimal("0.15"), ETH=Decimal("0.3"), )

        self.mocked_user_balances = MagicMock()
        self.mocked_user_balances._UserBalances__instance.all_balances.return_value = \
            dict(ALGO=Decimal("2000.8"), AVAX=Decimal("300.8"), BTC=Decimal("0.15"), ETH=Decimal("0.3"), )
        self.mocked_user_balances._UserBalances__instance.all_available_balances_all_exchanges.__getitem__ = \
            dict(ALGO=Decimal("1000.8"), AVAX=Decimal("300.8"), BTC=Decimal("0.15"), ETH=Decimal("0.3"), )

        self.fund_rebalancer = LiteStrategyFundRebalancer()

        self.expected_campaigns = \
            [{'Asset': 'ALGO', 'Exchange': 'kucoin', 'ETH Campaign': 1, 'ETH Skew': 50, 'USDT Campaign': 2.25,
              'USDT Skew': 30},
             {'Asset': 'AVAX', 'Exchange': 'kucoin', 'BTC Campaign': 1, 'BTC Skew': 50, 'USDT Campaign': 1,
              'USDT Skew': 50},
             ]
        self.expected_balances = \
            [['kucoin', 'ALGO', 2000.8, 1000.8, 10.0, 20008.0],
             ['kucoin', 'AVAX', 300.8, 300.8, 10.0, 3008.0],
             ['kucoin', 'BTC', 0.15, 0.15, 30000.0, 4500.0],
             ['kucoin', 'ETH', 0.3, 0.3, 1900.0, 570.0],
             ['kucoin', 'USDT', 0.0, 0.0, 1.0, 0.0],
             ]

        self.total_bal = {'ALGO': Decimal('2000.8'), 'AVAX': Decimal('300.8'), 'BTC': Decimal('0.15'),
                          'ETH': Decimal('0.3')}
        self.avail_bal = {'ALGO': Decimal('1000.8'), 'AVAX': Decimal('300.8'), 'BTC': Decimal('0.15'),
                          'ETH': Decimal('0.3')}
        self.prices = {'kucoin': {'ALGO-USDT': Decimal('10'),
                                  'AVAX-USDT': Decimal('10'),
                                  'ETH-USDT': Decimal('1900'),
                                  'BTC-USDT': Decimal('30000')}}

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    @staticmethod
    @patch('hummingbot.user.user_balances.UserBalances')
    def reset_mocking_one_asset(mocked_user_balances):
        mocked_user_balances._UserBalances__instance.all_balances.return_value = \
            dict(USDT=Decimal("1000"))
        mocked_user_balances._UserBalances__instance.all_available_balances_all_exchanges.__getitem__.return_value = \
            dict(USDT=Decimal("1000"))

        prices = {'kucoin': {"ETH-USDT": Decimal("1")}}
        market_init = MagicMock()
        market_init.base_currency = "USDT"
        market_init.hbot_weight = 1
        market_init.inventory_skew = 50
        market_init._config = \
            dict(
                markets=dict(kucoin=dict(quotes=dict(USDT={'ETH'}))),
                hbot_weight=1,
                inventory_skew=50,
                base_currency='USDT'
            )
        market_init.get_assets_from_config.return_value = {'USDT', 'ETH'}
        market_init.get_campaigns_from_config.return_value = market_init._config['markets']['kucoin']
        return mocked_user_balances, prices, market_init

    @staticmethod
    @patch('hummingbot.user.user_balances.UserBalances')
    def reset_mocking_two_assets(mocked_user_balances):
        mocked_user_balances._UserBalances__instance.all_balances.return_value = \
            dict(USDT=Decimal("1000"))
        mocked_user_balances._UserBalances__instance.all_available_balances_all_exchanges.__getitem__.return_value = \
            dict(USDT=Decimal("1000"))

        prices = {'kucoin': {"ETH-USDT": Decimal("1"), "ALGO-USDT": Decimal("0.1"), "ALGO-ETH": Decimal("0.1")}}
        market_init = MagicMock()
        market_init.base_currency = "USDT"
        market_init.hbot_weight = 1
        market_init.inventory_skew = 50
        market_init._config = \
            dict(
                markets=dict(kucoin=dict(quotes=dict(USDT={'ETH', 'ALGO'}),
                                         hbot={'ALGO-USDT'})),
                hbot_weight=1,
                inventory_skew=50,
                base_currency='USDT'
            )
        market_init.get_assets_from_config.return_value = {'USDT', 'ETH', 'ALGO'}
        market_init.get_campaigns_from_config.return_value = market_init._config['markets']['kucoin']
        return mocked_user_balances, prices, market_init

    def test__get_exchange_balances_w_connector(self):
        total, avail = self.fund_rebalancer._get_exchange_balances(connector=self.mocked_connector)
        self.assertEqual(total, self.total_bal)
        self.assertEqual(avail, self.avail_bal)

    @patch('hummingbot.user.user_balances.UserBalances')
    def test__get_exchange_balances_w_exchange(self, mocked_user_balances):
        mocked_user_balances._UserBalances__instance.all_balances.return_value = \
            dict(ALGO=Decimal("2000.8"), AVAX=Decimal("300.8"), BTC=Decimal("0.15"), ETH=Decimal("0.3"), )
        mocked_user_balances._UserBalances__instance.all_available_balances_all_exchanges.__getitem__.return_value = \
            dict(ALGO=Decimal("1000.8"), AVAX=Decimal("300.8"), BTC=Decimal("0.15"), ETH=Decimal("0.3"), )
        total, avail = self.fund_rebalancer._get_exchange_balances(exchange_name='kucoin')
        self.assertEqual(total, self.total_bal)
        self.assertEqual(avail, self.avail_bal)

    @patch('hummingbot.user.user_balances.UserBalances')
    def test__fetch_balances_prices_to_list(self, mocked_user_balances):
        exchange_name = "kucoin"
        mocked_user_balances._UserBalances__instance.all_balances.return_value = \
            dict(ALGO=Decimal("2000.8"), AVAX=Decimal("300.8"), BTC=Decimal("0.15"), ETH=Decimal("0.3"), )
        mocked_user_balances._UserBalances__instance.all_available_balances_all_exchanges.__getitem__.return_value = \
            dict(ALGO=Decimal("1000.8"), AVAX=Decimal("300.8"), BTC=Decimal("0.15"), ETH=Decimal("0.3"), )
        data = self.fund_rebalancer._fetch_balances_prices_to_list(self.market_init, self.prices,
                                                                   exchange_name=exchange_name)
        self.assertEqual(sorted(data), sorted(self.expected_balances))

        # Saving for later tests
        pd.Series(data).to_pickle("logs/data_balances.pickle")

    def test__reorganize_campaign_info_2quotes(self):
        exchange_name = "kucoin"

        data = self.fund_rebalancer._reorganize_campaign_info(self.market_init, exchange_name)
        self.assertEqual(sorted(data, key=lambda d: d['Asset']),
                         sorted(self.expected_campaigns, key=lambda d: d['Asset']))

        # Saving for later tests
        pd.Series(data).to_pickle('logs/data_campaigns.pickle')

    def test__combine_assets_campaigns(self):
        try:
            balances = pd.read_pickle('logs/data_balances.pickle').tolist()
        except (IOError, EOFError):
            balances = self.expected_balances

        try:
            campaigns = sorted(pd.read_pickle('logs/data_campaigns.pickle').tolist(), key=lambda d: d['Asset'])
        except (IOError, EOFError):
            campaigns = self.expected_campaigns

        data = self.fund_rebalancer._combine_assets_campaigns(balances, campaigns)

        exp_columns = sorted(
            ["Exchange", "Asset", "Balance", "Available Balance", "Price (USDT)", "Capital (USDT)",
             'ETH Campaign', 'USDT Campaign', 'BTC Campaign', 'ETH Skew', 'USDT Skew', 'BTC Skew'])
        data_columns = sorted(data.columns.values)

        self.assertEqual(data_columns, exp_columns)

        # Saving for later tests
        data.to_pickle("logs/data_data_df.pickle")

    def test__calculate_funds(self):
        try:
            input_df = pd.read_pickle('logs/data_data_df.pickle')
        except (IOError, EOFError):
            print("Run test__combine_assets_campaigns to create a valid input DataFrame for this test")

        data = self.fund_rebalancer._calculate_funds(input_df)

        exp_columns = sorted(
            ["Exchange", "Capital (USDT)", 'ETH Campaign', 'USDT Campaign', 'BTC Campaign', 'Total Campaigns',
             'ETH Allocation (USDT)', 'USDT Allocation (USDT)', 'BTC Allocation (USDT)',
             'Unit Funding Base+Quote (USDT)'])
        data_columns = sorted(data.columns.values)

        test_unit_fund = (input_df['Capital (USDT)'].sum()) / (input_df.filter(regex=r"\w+ Campaign$").sum().sum())

        self.assertEqual(data_columns, exp_columns)
        self.assertEqual(data['Unit Funding Base+Quote (USDT)'].values[0], test_unit_fund)

        # Saving for later tests
        data.to_pickle("logs/data_funds_df.pickle")

    def test__calculate_asset_funding_per_exchange(self):
        data_df = pd.read_pickle('logs/data_data_df.pickle')
        funds_df = pd.read_pickle('logs/data_funds_df.pickle')

        data = self.fund_rebalancer._calculate_asset_funding_per_exchange(data_df, 'Exchange', funds_df)

        exp_columns = sorted(
            ['Asset', 'Available Balance', 'Capital (USDT)', 'ETH Campaign', 'Total ETH Funds (USDT)', 'ETH Skew',
             'BTC Campaign', 'Total BTC Funds (USDT)', 'BTC Skew', 'Exchange', 'Price (USDT)', 'Total Funding (USDT)',
             'Balance', 'USDT Campaign',
             'Total USDT Funds (USDT)', 'USDT Skew', 'Total Funding Gap (USDT)'])
        data_columns = sorted(data.columns.values)

        test_unit_fund = data_df[data_df['Exchange'] == 'kucoin'][['USDT Campaign']] * funds_df[funds_df['Exchange'] == 'kucoin']['Unit Funding Base+Quote (USDT)'].values

        self.assertEqual(data_columns, exp_columns)
        self.assertEqual(sorted(data[['Total USDT Funds (USDT)']].values),
                         sorted(test_unit_fund.values))

        # Saving for later tests
        data.to_pickle("logs/data_asset_funds_df.pickle")

    def test__calculate_base_quote_ratio(self):
        data_df = pd.read_pickle('logs/data_asset_funds_df.pickle')

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            data = self.fund_rebalancer._calculate_base_quote_ratio(data_df)

        exp_columns = sorted(['Exchange', 'Asset', 'Balance', 'Available Balance', 'Price (USDT)',
                              'Capital (USDT)', 'USDT Campaign', 'USDT Skew', 'ETH Campaign',
                              'ETH Skew', 'BTC Campaign', 'BTC Skew', 'Total Funding (USDT)',
                              'Total Funding Gap (USDT)', 'Total USDT Funds (USDT)',
                              'Total ETH Funds (USDT)', 'Total BTC Funds (USDT)',
                              'Base USDT Funds (USDT)', 'Base ETH Funds (USDT)',
                              'Base BTC Funds (USDT)', 'Base Funding (USDT)',
                              'Quote USDT Funds (USDT)', 'Quote ETH Funds (USDT)',
                              'Quote BTC Funds (USDT)', 'Funding Gap (USDT)', 'Funding Gap (asset)',
                              'Quote Funding (USDT)'])
        data_columns = sorted(data.columns.values)

        self.assertEqual(data_columns, exp_columns)

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #    print(data)

        # Saving for later tests
        data.to_pickle("logs/data_ratio_asset_funds_df.pickle")

    def test__compose_sells(self):
        data_df = pd.read_pickle('logs/data_ratio_asset_funds_df.pickle')

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            data = self.fund_rebalancer._compose_sells(data_df)

        expected = [{'amount': 11100.657142857142, 'asset': 'ALGO', 'to': 'USDT'},
                    {'amount': 2341.7142857142853, 'asset': 'ALGO', 'to': 'AVAX'},
                    {'amount': 1825.1428571428573, 'asset': 'BTC', 'to': 'ETH'},
                    {'amount': 279.7142857142853, 'asset': 'ALGO', 'to': 'ETH'}]

        self.assertEqual(data, expected)

    @patch('hummingbot.user.user_balances.UserBalances')
    def test_rebalancing_proposal_one_asset(self, mocked_user_balances):

        # One asset without inventory, price of 1 no modificators, 50-50
        mocked_user_balances, prices, market_init = self.reset_mocking_one_asset(mocked_user_balances)
        expected = [{'amount': 500.0, 'asset': 'USDT', 'to': 'ETH'}]
        data = self.fund_rebalancer.rebalancing_proposal(market_init, prices,
                                                         connectors={'kucoin': self.mocked_connector})
        diff = [i for i in data + expected if i not in data or i not in expected]
        self.assertEqual(len(diff), 0)

        # One asset without inventory, price of 1, inventory skew, 70-30
        mocked_user_balances, prices, market_init = self.reset_mocking_one_asset(mocked_user_balances)
        market_init.inventory_skew = 30
        expected = [{'amount': 300.0, 'asset': 'USDT', 'to': 'ETH'}]
        data = self.fund_rebalancer.rebalancing_proposal(market_init, prices,
                                                         connectors={'kucoin': self.mocked_connector})
        diff = [i for i in data + expected if i not in data or i not in expected]
        self.assertEqual(len(diff), 0)

        # One asset without inventory, price of 1, inventory skew, 50-50, base currency ETH
        mocked_user_balances, prices, market_init = self.reset_mocking_one_asset(mocked_user_balances)
        market_init.base_currency = 'ETH'
        expected = [{'amount': 500.0, 'asset': 'USDT', 'to': 'ETH'}]
        data = self.fund_rebalancer.rebalancing_proposal(market_init, prices,
                                                         connectors={'kucoin': self.mocked_connector})
        diff = [i for i in data + expected if i not in data or i not in expected]
        self.assertEqual(diff, [])

        # ValueError raised if an asset has no price in the base currency
        mocked_user_balances, prices, market_init = self.reset_mocking_one_asset(mocked_user_balances)
        market_init.get_assets_from_config.return_value = {'USDT', 'ALGO'}
        market_init._config['markets']['kucoin']['quotes'] = dict(USDT={'ALGO'})
        with self.assertRaises(ValueError):
            self.fund_rebalancer.rebalancing_proposal(market_init, prices,
                                                      connectors={'kucoin': self.mocked_connector})

    @patch('hummingbot.user.user_balances.UserBalances')
    def test_rebalancing_proposal_two_asset_default(self, mocked_user_balances):
        # Two assets without inventory, price of 1 no modificators, 50-50
        mocked_user_balances, prices, market_init = self.reset_mocking_two_assets(mocked_user_balances)
        # 50-50, thus 50% USDT, 50% ETH and ALGO
        expected = [{'amount': 250.0, 'asset': 'USDT', 'to': 'ETH'},
                    {'amount': 250.0, 'asset': 'USDT', 'to': 'ALGO'}]
        data = self.fund_rebalancer.rebalancing_proposal(market_init, prices,
                                                         connectors={'kucoin': self.mocked_connector})
        diff = [i for i in data + expected if i not in data or i not in expected]
        self.assertEqual(diff, [])

    def test_rebalancing_proposal_two_asset_skew(self, mocked_user_balances):
        # Two assets without inventory, price of 1, inventory skew, 70-30
        mocked_user_balances, prices, market_init = self.reset_mocking_two_assets(mocked_user_balances)
        market_init.inventory_skew = 30
        # 70-30, thus 70% USDT, 30% ETH and ALGO
        expected = [{'amount': 150.0, 'asset': 'USDT', 'to': 'ETH'},
                    {'amount': 150.0, 'asset': 'USDT', 'to': 'ALGO'}]
        data = self.fund_rebalancer.rebalancing_proposal(market_init, prices,
                                                         connectors={'kucoin': self.mocked_connector})
        diff = [i for i in data + expected if i not in data or i not in expected]
        self.assertEqual(diff, [])

    def test_rebalancing_proposal_two_asset_specific_skew(self, mocked_user_balances):
        # Two assets without inventory, price of 1, specific inventory skew, 70-30
        mocked_user_balances, prices, market_init = self.reset_mocking_two_assets(mocked_user_balances)
        market_init._config['markets']['kucoin']['inventory_skews'] = {"ALGO-USDT": 30}
        # 50-50 ETH and 70-30 ALGO
        expected = [{'amount': 250.0, 'asset': 'USDT', 'to': 'ETH'},
                    {'amount': 150.0, 'asset': 'USDT', 'to': 'ALGO'}]
        data = self.fund_rebalancer.rebalancing_proposal(market_init, prices,
                                                         connectors={'kucoin': self.mocked_connector})
        diff = [i for i in data + expected if i not in data or i not in expected]
        self.assertEqual(diff, [])

    def test_rebalancing_proposal_two_asset_base_currency(self, mocked_user_balances):
        # Two assets without inventory, price of 1, inventory skew, 50-50, base currency ETH
        mocked_user_balances, prices, market_init = self.reset_mocking_two_assets(mocked_user_balances)
        market_init.base_currency = 'ETH'
        # No change, there is an error if an asset does not have a ETH price
        expected = [{'amount': 250.0, 'asset': 'USDT', 'to': 'ETH'},
                    {'amount': 250.0, 'asset': 'USDT', 'to': 'ALGO'}]
        data = self.fund_rebalancer.rebalancing_proposal(market_init, prices,
                                                         connectors={'kucoin': self.mocked_connector})
        diff = [i for i in data + expected if i not in data or i not in expected]
        self.assertEqual(diff, [])

    def test_rebalancing_proposal_two_asset_hbot(self, mocked_user_balances):
        # Two assets without inventory, price of 1, inventory skew, 50-50, hbot_weight 1.5x
        mocked_user_balances, prices, market_init = self.reset_mocking_two_assets(mocked_user_balances)
        market_init.hbot_weight = 1.5
        # 50-500, thus 50% USDT, but different ALGO is 1.5x ETH using global hbot weight
        expected = [{'amount': 300.0, 'asset': 'USDT', 'to': 'ALGO'},
                    {'amount': 200.0, 'asset': 'USDT', 'to': 'ETH'}]
        data = self.fund_rebalancer.rebalancing_proposal(market_init, prices,
                                                         connectors={'kucoin': self.mocked_connector})
        diff = [i for i in data + expected if i not in data or i not in expected]
        self.assertEqual(diff, [])

    def test_rebalancing_proposal_two_asset_specific_hbot(self, mocked_user_balances):
        # Two assets without inventory, price of 1, inventory skew, 50-50, specific hbot_weights 1.5x
        mocked_user_balances, prices, market_init = self.reset_mocking_two_assets(mocked_user_balances)
        market_init._config['markets']['kucoin']['hbot_weights'] = {"ALGO-USDT": 1.5}
        # 50-500, thus 50% USDT, but different ALGO is 1.5x ETH using global hbot weight
        expected = [{'amount': 300.0, 'asset': 'USDT', 'to': 'ALGO'},
                    {'amount': 200.0, 'asset': 'USDT', 'to': 'ETH'}]
        data = self.fund_rebalancer.rebalancing_proposal(market_init, prices,
                                                         connectors={'kucoin': self.mocked_connector})
        diff = [i for i in data + expected if i not in data or i not in expected]
        self.assertEqual(diff, [])

    @patch('hummingbot.user.user_balances.UserBalances')
    def test_rebalancing_proposal(self, mocked_user_balances):
        mocked_user_balances._UserBalances__instance.all_balances.return_value = \
            dict(ALGO=Decimal("2000.8"), AVAX=Decimal("300.8"), BTC=Decimal("0.15"), ETH=Decimal("0.3"), )
        mocked_user_balances._UserBalances__instance.all_available_balances_all_exchanges.__getitem__.return_value = \
            dict(ALGO=Decimal("1000.8"), AVAX=Decimal("300.8"), BTC=Decimal("0.15"), ETH=Decimal("0.3"), )

        expected = [{'amount': 11100.657142857142, 'asset': 'ALGO', 'to': 'USDT'},
                    {'amount': 2341.7142857142853, 'asset': 'ALGO', 'to': 'AVAX'},
                    {'amount': 1825.1428571428573, 'asset': 'BTC', 'to': 'ETH'},
                    {'amount': 279.7142857142853, 'asset': 'ALGO', 'to': 'ETH'}]
        data = self.fund_rebalancer.rebalancing_proposal(self.market_init, self.prices,
                                                         connectors={'kucoin': self.mocked_connector})
        self.assertEqual(data, expected)
