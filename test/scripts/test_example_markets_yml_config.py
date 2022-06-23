import unittest
from unittest.mock import call, patch

from scripts.example_markets_yml_config import ExampleMarketsYmlConfig
from scripts.markets_yml_config import MarketsYmlConfig


class TestExampleMarketsYmlConfig(unittest.TestCase):

    def test_initialize_from_yml_config(self):
        # Test could fail due to class attribute being persistent
        self.assertEqual(ExampleMarketsYmlConfig.config_filename, 'conf/conf_ls_config_file.yml')

    def test_initialize_from_yml_wo_markets(self):
        # Need to do this due to persistence of class members (random order of test)
        if hasattr(ExampleMarketsYmlConfig, 'markets'):
            delattr(ExampleMarketsYmlConfig, 'markets')
        with patch.object(MarketsYmlConfig, 'load_from_yml') as mocked_load:
            with patch.object(MarketsYmlConfig, 'update_markets') as mocked_update:
                with patch.object(MarketsYmlConfig, 'initialize_markets') as mocked_init:
                    mocked_init.return_value = {'kucoin': {'ALGO-ETH', 'ALGO-USDT', 'AVAX-USDT', 'AVAX-BTC'}}
                    markets = ExampleMarketsYmlConfig.initialize_from_yml()

        self.assertEqual(mocked_load.call_args_list, [call('conf/conf_ls_config_file.yml')])
        self.assertEqual(mocked_update.call_args_list, [])
        self.assertEqual(mocked_init.call_args_list, [call('conf/conf_ls_config_file.yml')])
        self.assertEqual(markets, {'kucoin': {'ALGO-ETH', 'ALGO-USDT', 'AVAX-USDT', 'AVAX-BTC'}})

    def test_initialize_from_yml_w_markets(self):
        with patch.object(MarketsYmlConfig, 'load_from_yml') as mocked_load:
            ExampleMarketsYmlConfig.markets = {'kucoin': {'ALGO-ETH', 'ALGO-USDT', 'AVAX-USDT', 'AVAX-BTC'}}
            with patch.object(MarketsYmlConfig, 'update_markets') as mocked_update:
                with patch.object(MarketsYmlConfig, 'initialize_markets') as mocked_init:
                    mocked_init.return_value = {'kucoin': {'ALGO-ETH', 'ALGO-USDT', 'AVAX-USDT', 'AVAX-BTC'}}
                    markets = ExampleMarketsYmlConfig.initialize_from_yml()

        self.assertEqual(mocked_load.call_args_list, [call('conf/conf_ls_config_file.yml')])
        self.assertEqual(mocked_update.call_args_list,
                         [call({'kucoin': {'ALGO-ETH', 'AVAX-USDT', 'AVAX-BTC', 'ALGO-USDT'}})])
        self.assertEqual(mocked_init.call_args_list, [call('conf/conf_ls_config_file.yml')])
        self.assertEqual(markets, {'kucoin': {'ALGO-ETH', 'ALGO-USDT', 'AVAX-USDT', 'AVAX-BTC'}})
