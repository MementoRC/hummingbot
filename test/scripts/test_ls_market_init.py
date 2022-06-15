import unittest
from unittest.mock import MagicMock, call, mock_open, patch

import ruamel.yaml

from scripts.ls_markets_init import LiteStrategyMarketsInit

yaml_parser = ruamel.yaml.YAML()


class TestLiteStrategyMarketsInit(unittest.TestCase):
    def setUp(self) -> None:
        self._config = dict(markets=dict(kucoin=dict(quotes=dict(USDT={'ALGO', 'AVAX'}, ETH={'ALGO'}, BTC={'AVAX'}),
                                                     weights={'ALGO-USDT': 1.5},
                                                     hbot=set(['ALGO-USDT']),
                                                     inventory_skews={"ALGO-USDT": 30},
                                                     hbot_weights={"ALGO-USDT": 1.5})),
                            hbot_weight=1,
                            inventory_skew=50,
                            base_currency='USDT'
                            )

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_initialize_markets(self):
        file_name = "conf/conf_ls_markets_init.yml"

        # Initialization of class to test with valid _config
        LiteStrategyMarketsInit._save_config_to_yml = MagicMock()
        LiteStrategyMarketsInit._config = dict(
            markets=dict(kucoin=dict(quotes=dict(USDT={'ALGO', 'AVAX'}, ETH={'ALGO'}, BTC={'AVAX'}),
                                     weights={'ALGO-USDT': 1.5},
                                     hbot={'ALGO-USDT'},
                                     inventory_skews={"ALGO-USDT": 30},
                                     hbot_weights={"ALGO-USDT": 1.5})),
            hbot_weight=1,
            inventory_skew=50,
            base_currency='USDT')

        # Expected markets based on _config['markets']
        yml_markets = {'kucoin': {'ALGO-ETH', 'ALGO-USDT', 'AVAX-USDT', 'AVAX-BTC'}}
        # The quote-base_currency are automatically added to facilitate re-balancing quotes
        yml_markets['kucoin'].add('ETH-USDT')
        yml_markets['kucoin'].add('BTC-USDT')

        markets = LiteStrategyMarketsInit.initialize_markets(file_name)

        self.assertEqual(LiteStrategyMarketsInit._save_config_to_yml.call_args, call('conf/conf_ls_markets_init.yml'))
        self.assertEqual(yml_markets, markets)

    def test_update_markets(self):
        # Class defined markets input
        markets = {'kucoin': {'ALGO-ETH', 'ALGO-USDT', 'AVAX-USDT', 'AVAX-BTC'}}
        exp_config = dict(
            markets=dict(kucoin=dict(assets={'AVAX', 'ALGO'},
                                     quotes=dict(USDT={'ALGO', 'AVAX'}, ETH={'ALGO'}, BTC={'AVAX'}),
                                     weights=dict(),
                                     hbot=set(),
                                     inventory_skews=dict(),
                                     hbot_weights=dict())),
            hbot_weight=1,
            inventory_skew=50,
            base_currency='USDT')

        # Initialized class with a sample yml config load different from the markets
        LiteStrategyMarketsInit._config = dict(
            markets=dict(kucoin=dict(quotes=dict(USDT={'BTC', 'DAO'}, ETH={'BTC'}, BTC={'DAO'}),
                                     weights={'BTC-USDT': 1.5},
                                     hbot={'DAO-USDT'},
                                     inventory_skews={"BTC-USDT": 30},
                                     hbot_weights={"BTC-USDT": 1.5})),
            hbot_weight=1,
            inventory_skew=50,
            base_currency='USDT')

        LiteStrategyMarketsInit.update_markets(markets)

        self.assertEqual(exp_config, LiteStrategyMarketsInit._config, )

    def test_load_from_yml_no_file(self):
        # Class defined markets input
        exp_config = {'base_currency': 'USDT',
                      'hbot_weight': 1,
                      'inventory_skew': 50,
                      'markets': {'gate_io': {'assets': set(),
                                              'hbot': {'XCAD-USDT'},
                                              'hbot_weights': {'XCAD-USDT': 1.5},
                                              'inventory_skews': {},
                                              'quotes': {'ETH': ['VSP'],
                                                         'USDT': ['XCAD']},
                                              'weights': {'XCAD-USDT': 1.5}},
                                  'kucoin': {'assets': set(),
                                             'hbot': {'XCAD-USDT'},
                                             'hbot_weights': {'XCAD-USDT': 1.5},
                                             'inventory_skews': {'XCAD-USDT': 30},
                                             'quotes': {'BTC': ['VID'],
                                                        'ETH': ['NIM'],
                                                        'USDT': ['XCAD']},
                                             'weights': {'XCAD-USDT': 1.5}}}}

        mocking_isfile = MagicMock()
        mocking_isfile.return_value = False
        with patch('scripts.ls_markets_init.isfile', mocking_isfile):
            LiteStrategyMarketsInit.load_from_yml('conf/conf_ls_markets_init.yml')

        self.assertEqual(exp_config, LiteStrategyMarketsInit._config, )

    def test_load_from_yml_config(self):
        # Class defined markets input
        exp_config = dict(
            markets=dict(kucoin=dict(assets={'AVAX', 'ALGO'},
                                     quotes=dict(USDT={'ALGO', 'AVAX'}, ETH={'ALGO'}, BTC={'AVAX'}),
                                     weights=dict(),
                                     hbot=set(),
                                     inventory_skews=dict(),
                                     hbot_weights=dict())),
            hbot_weight=1,
            inventory_skew=50,
            base_currency='USDT')

        mocking_isfile = MagicMock()
        mocking_isfile.return_value = True
        with patch('scripts.ls_markets_init.isfile', mocking_isfile):
            with patch('builtins.open', mock_open()) as mocked_file:
                with patch('ruamel.yaml.YAML.load') as mocked_load:
                    mocked_load.return_value = exp_config
                    LiteStrategyMarketsInit.load_from_yml('conf/conf_ls_markets_init.yml')

        self.assertEqual(mocked_load.call_args, call(mocked_file()))
        self.assertEqual(exp_config, LiteStrategyMarketsInit._config)

    def test__save_config_to_yml_no_config(self):
        mocking_open = mock_open()
        with patch('builtins.open', mocking_open):
            LiteStrategyMarketsInit._save_config_to_yml('conf/conf_ls_markets_init.yml')
        # Expecting no calls to open, since no _config to save
        self.assertEqual(mocking_open.call_args_list, [])

    def test__save_config_to_yml_open(self):
        LiteStrategyMarketsInit._config = self._config
        mocking_open = mock_open()
        with patch('builtins.open', mocking_open):
            LiteStrategyMarketsInit._save_config_to_yml('conf/conf_ls_markets_init.yml')
        # Expecting two calls to open, one that reads as a stream, second that writes to the file
        self.assertEqual(mocking_open.call_args_list, [call('conf/conf_ls_markets_init.yml', 'r'),
                                                       call('conf/conf_ls_markets_init.yml', 'w+'),
                                                       ])

    def test_save_markets_to_yml_yaml(self):
        LiteStrategyMarketsInit._config = self._config
        with patch('builtins.open', mock_open()) as mocked_file:
            with patch('ruamel.yaml.YAML.load') as mocked_load:
                with patch('ruamel.yaml.YAML.dump') as mocked_dump:
                    LiteStrategyMarketsInit._save_config_to_yml('conf/conf_ls_markets_init.yml')
        self.assertEqual(mocked_load.call_args, call(mocked_file()))
        self.assertEqual(mocked_dump.call_args, call(mocked_load(), mocked_file()))
