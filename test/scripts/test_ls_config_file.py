import unittest
from unittest.mock import call, patch

from scripts.ls_config_file import LiteStrategyConfigFile
from scripts.ls_markets_init import LiteStrategyMarketsInit


class LiteStrategyConfigFileTest(unittest.TestCase):

    def test_initialize_from_yml_config(self):
        # Test could fail due to class attribute being persistent
        self.assertEqual(LiteStrategyConfigFile.config_filename, 'conf/conf_ls_config_file.yml')

    def test_initialize_from_yml_wo_markets(self):
        # Need to do this due to persistence of class members (random order of test)
        if hasattr(LiteStrategyConfigFile, 'markets'):
            delattr(LiteStrategyConfigFile, 'markets')
        with patch.object(LiteStrategyMarketsInit, 'load_from_yml') as mocked_load:
            with patch.object(LiteStrategyMarketsInit, 'update_markets') as mocked_update:
                with patch.object(LiteStrategyMarketsInit, 'initialize_markets') as mocked_init:
                    mocked_init.return_value = {'kucoin': {'ALGO-ETH', 'ALGO-USDT', 'AVAX-USDT', 'AVAX-BTC'}}
                    markets = LiteStrategyConfigFile.initialize_from_yml()

        self.assertEqual(mocked_load.call_args_list, [call('conf/conf_ls_config_file.yml')])
        self.assertEqual(mocked_update.call_args_list, [])
        self.assertEqual(mocked_init.call_args_list, [call('conf/conf_ls_config_file.yml')])
        self.assertEqual(markets, {'kucoin': {'ALGO-ETH', 'ALGO-USDT', 'AVAX-USDT', 'AVAX-BTC'}})

    def test_initialize_from_yml_w_markets(self):
        with patch.object(LiteStrategyMarketsInit, 'load_from_yml') as mocked_load:
            LiteStrategyConfigFile.markets = {'kucoin': {'ALGO-ETH', 'ALGO-USDT', 'AVAX-USDT', 'AVAX-BTC'}}
            with patch.object(LiteStrategyMarketsInit, 'update_markets') as mocked_update:
                with patch.object(LiteStrategyMarketsInit, 'initialize_markets') as mocked_init:
                    mocked_init.return_value = {'kucoin': {'ALGO-ETH', 'ALGO-USDT', 'AVAX-USDT', 'AVAX-BTC'}}
                    markets = LiteStrategyConfigFile.initialize_from_yml()

        self.assertEqual(mocked_load.call_args_list, [call('conf/conf_ls_config_file.yml')])
        self.assertEqual(mocked_update.call_args_list,
                         [call({'kucoin': {'ALGO-ETH', 'AVAX-USDT', 'AVAX-BTC', 'ALGO-USDT'}})])
        self.assertEqual(mocked_init.call_args_list, [call('conf/conf_ls_config_file.yml')])
        self.assertEqual(markets, {'kucoin': {'ALGO-ETH', 'ALGO-USDT', 'AVAX-USDT', 'AVAX-BTC'}})
