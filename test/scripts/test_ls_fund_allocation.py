import unittest
from decimal import Decimal
from unittest.mock import MagicMock, call, mock_open, patch

import pandas as pd
import ruamel.yaml

from hummingbot.connector.mock.mock_paper_exchange import MockPaperExchange
from scripts.report_balance_status import ReportBalanceStatus

yaml_parser = ruamel.yaml.YAML()


# TODO: Add test for async calls on the main event loop #noqa
# TODO: Rename class to correct example #noqa

class TestReportBalanceStatus(unittest.TestCase):
    def setUp(self) -> None:
        self.log_records = []
        self.connector_name: str = "mock_paper_exchange"
        self.trading_pair: str = "HBOT-USDT"
        self.base_asset, self.quote_asset = self.trading_pair.split("-")
        self.base_balance: int = 500
        self.quote_balance: int = 5000
        self.initial_mid_price: int = 100
        self.connector: MockPaperExchange = MockPaperExchange()
        self.connector.set_balanced_order_book(trading_pair=self.trading_pair,
                                               mid_price=100,
                                               min_price=50,
                                               max_price=150,
                                               price_step_size=1,
                                               volume_step_size=10)
        self.connector.set_balance(self.base_asset, self.base_balance)
        self.connector.set_balance(self.quote_asset, self.quote_balance)
        ReportBalanceStatus.initialize_markets(dict())
        self.strategy = ReportBalanceStatus({self.connector_name: self.connector})
        self.strategy.logger().setLevel(1)
        self.strategy.logger().addHandler(self)
        self.strategy._config = MagicMock()
        self.strategy._config = \
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
        self.expected_campaigns = \
            [{'Asset': 'ALGO', 'Exchange': 'kucoin', 'ETH Campaign': 1, 'ETH Skew': 50, 'USDT Campaign': 2.25,
              'USDT Skew': 30},
             {'Asset': 'AVAX', 'Exchange': 'kucoin', 'BTC Campaign': 1, 'BTC Skew': 50, 'USDT Campaign': 1,
              'USDT Skew': 50},
             ]
        self.expected_balances = \
            [['kucoin', 'ALGO', 2000.8, 1000.8, 10, 20008],
             ['kucoin', 'AVAX', 300.8, 300.8, 10, 3008],
             ['kucoin', 'BTC', 0.15, 0.15, 30000, 4500],
             ['kucoin', 'ETH', 0, 0, 1900, 0],
             ['kucoin', 'USDT', 0, 0, 1, 0],
             ]
        self.strategy.get_strategy_exchange_assets = MagicMock()
        self.strategy.get_exchange_balances = MagicMock()
        self.strategy._prices = MagicMock()

        self.strategy.get_strategy_exchange_assets.return_value = {'ALGO', 'AVAX', 'BTC', 'USDT', 'ETH'}
        self.strategy.get_exchange_balances.return_value = ({'ALGO': Decimal('2000.8'),
                                                             'AVAX': Decimal('300.8'),
                                                             'BTC': Decimal('0.15')},
                                                            {'ALGO': Decimal('1000.8'),
                                                             'AVAX': Decimal('300.8'),
                                                             'BTC': Decimal('0.15')})
        self.strategy._prices = {'kucoin': {'ALGO-USDT': Decimal('10'),
                                            'AVAX-USDT': Decimal('10'),
                                            'ETH-USDT': Decimal('1900'),
                                            'BTC-USDT': Decimal('30000')
                                            }}

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_initialize_markets_file_name(self):
        strategy_name = "report_balance_status"
        file_name = f"conf/conf_{strategy_name}.yml"

        markets = dict()
        yml_markets = {'kucoin': {'NIM-ETH', 'ALGO-USDT', 'AVAX-USDT', 'ALGO-BTC'},
                       'gate_io': {'FEAR-USDT', 'HMT-ETH', 'AVAX-USDT'}}

        # The quote-base_currency are added to facilitate re-balancing quotes
        yml_markets['kucoin'].add('ETH-USDT')
        yml_markets['kucoin'].add('BTC-USDT')
        yml_markets['gate_io'].add('ETH-USDT')

        ReportBalanceStatus.initialize_markets(markets)

        self.assertEqual(file_name, ReportBalanceStatus.config_filename)
        self.assertEqual(yml_markets, markets)

    def test_save_markets_to_yml_open(self):
        mocking_open = mock_open()
        with patch('scripts.report_balance_status.open', mocking_open):
            ReportBalanceStatus.save_markets_to_yml()
        self.assertEqual(mocking_open.call_args_list, [call('conf/conf_report_balance_status.yml', 'r'),
                                                       call('conf/conf_report_balance_status.yml', 'w+'),
                                                       ])

    def test_save_markets_to_yml_yaml(self):
        with patch('scripts.report_balance_status.open', mock_open()) as mocked_file:
            with patch('scripts.report_balance_status.ruamel.yaml.YAML.load') as mocked_load:
                with patch('scripts.report_balance_status.ruamel.yaml.YAML.dump') as mocked_dump:
                    self.strategy.save_markets_to_yml()
        self.assertEqual(mocked_load.call_args, call(mocked_file()))
        self.assertEqual(mocked_dump.call_args, call(mocked_load(), mocked_file()))

    @patch('hummingbot.connector.connector_base.ConnectorBase')
    def test_get_exchange_balance_connector(self, mock_class):
        mock_class.get_all_balances.return_value = dict(
            ALGO=Decimal(2000),
            ETH=Decimal(0.3),
            AVAX=Decimal(7),
            BTC=Decimal(0.05)
        )
        mock_class.available_balances = dict(
            ALGO=Decimal(2000),
            ETH=Decimal(0.3),
            AVAX=Decimal(7),
            BTC=Decimal(0.05)
        )
        total, avail = self.strategy.get_exchange_balances(connector=mock_class)
        self.assertEqual(total['ETH'], 0.3)
        self.assertEqual(avail['BTC'], 0.05)

    @patch('hummingbot.user.user_balances.UserBalances')
    def test_get_exchange_balance_exchange(self, mock_class):
        mock_class._UserBalances__instance.all_balances.return_value = dict(
            ALGO=Decimal(2000),
            ETH=Decimal(0.3),
            AVAX=Decimal(7),
            BTC=Decimal(0.05)
        )
        mock_class._UserBalances__instance.all_available_balances.return_value = dict(
            ALGO=Decimal(2000),
            ETH=Decimal(0.3),
            AVAX=Decimal(7),
            BTC=Decimal(0.05)
        )
        total, avail = self.strategy.get_exchange_balances(exchange_name='kucoin')
        self.assertEqual(total['ETH'], 0.3)
        self.assertEqual(avail['BTC'], 0.05)

    def test__fetch_balances_prices(self):
        exchange_name = "kucoin"

        data = self.strategy._fetch_balances_prices(exchange_name)
        self.assertEqual(sorted(data), sorted(self.expected_balances))

        # Saving for later tests
        pd.Series(data).to_pickle("balances.pickle")

    def test__reorganize_campaign_info_2quotes(self):
        exchange_name = "kucoin"

        data = self.strategy._reorganize_campaign_info(exchange_name)
        self.assertEqual(sorted(data, key=lambda d: d['Asset']),
                         sorted(self.expected_campaigns, key=lambda d: d['Asset']))

        # Saving for later tests
        pd.Series(data).to_pickle('campaigns.pickle')

    def test__combine_assets_campaigns(self):
        try:
            balances = pd.read_pickle('balances.pickle').tolist()
        except (IOError, EOFError):
            balances = self.expected_balances

        try:
            campaigns = sorted(pd.read_pickle('campaigns.pickle').tolist(), key=lambda d: d['Asset'])
        except (IOError, EOFError):
            campaigns = self.expected_campaigns

        data = self.strategy._combine_assets_campaigns(balances, campaigns)

        exp_columns = sorted(
            ["Exchange", "Asset", "Balance", "Available Balance", "Price (USDT)", "Capital (USDT)",
             'ETH Campaign', 'USDT Campaign', 'BTC Campaign', 'ETH Skew', 'USDT Skew', 'BTC Skew'])
        data_columns = sorted(data.columns.values)

        self.assertEqual(data_columns, exp_columns)

        # Saving for later tests
        data.to_pickle("data_df.pickle")

    def test__calculate_funds(self):
        try:
            input_df = pd.read_pickle('data_df.pickle')
        except (IOError, EOFError):
            print("Run test__combine_assets_campaigns to create a valid input DataFrame for this test")

        data = self.strategy._calculate_funds(input_df)

        exp_columns = sorted(
            ["Exchange", "Capital (USDT)", 'ETH Campaign', 'USDT Campaign', 'BTC Campaign', 'Total Campaigns',
             'ETH Allocation (USDT)', 'USDT Allocation (USDT)', 'BTC Allocation (USDT)',
             'Unit Funding Base+Quote (USDT)'])
        data_columns = sorted(data.columns.values)

        test_unit_fund = (input_df['Capital (USDT)'].sum()) / (input_df.filter(regex=r"\w+ Campaign$").sum().sum())

        self.assertEqual(data_columns, exp_columns)
        self.assertEqual(data['Unit Funding Base+Quote (USDT)'].values[0], test_unit_fund)

        # Saving for later tests
        data.to_pickle("funds_df.pickle")

    def test__calculate_asset_funding_per_exchange(self):
        data_df = pd.read_pickle('data_df.pickle')
        funds_df = pd.read_pickle('funds_df.pickle')

        data = self.strategy._calculate_asset_funding_per_exchange(data_df, 'Exchange', funds_df)

        print(data)
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
        data.to_pickle("asset_funds_df.pickle")

    def test__calculate_base_quote_ratio(self):
        data_df = pd.read_pickle('asset_funds_df.pickle')

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            data = self.strategy._calculate_base_quote_ratio(data_df)

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
        data.to_pickle("ratio_asset_funds_df.pickle")

    def test__implement_buys_sells(self):
        data_df = pd.read_pickle('ratio_asset_funds_df.pickle')

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            data = self.strategy._implement_buys_sells(data_df)

        expected = [{'asset': 'ALGO', 'to': 'USDT', 'amount': 10875.371428571427},
                    {'asset': 'ALGO', 'to': 'ETH', 'amount': 2620.5714285714284},
                    {'asset': 'BTC', 'to': 'AVAX', 'amount': 1879.4285714285716},
                    {'asset': 'ALGO', 'to': 'AVAX', 'amount': 353.7142857142853}]

        self.assertEqual(data, expected)
