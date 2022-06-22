from os.path import isfile
from typing import Dict, Set, Union

import ruamel.yaml

yaml_parser = ruamel.yaml.YAML()


class LiteStrategyMarketsInit(object):
    """
    Testing a market initialization method for Script Strategies
    Initialize the list of markets prior the market initialization phase in Hummingbot
    """
    HMarkets = Dict[str, Union[Dict[str, Union[Set, Dict]], Dict[str, Union[Set, Dict]]]]
    Config = Dict[str, Union[str, int, HMarkets, Set[str]]]

    _config: Config

    @classmethod
    def load_from_yml(cls, config_filename: str) -> None:
        """
        Initializes the _config class attribute with yml config file

        :param config_filename: name of the configuration file
        """
        # Does a configuration file exists? If so we load it in a class member
        if isfile(config_filename):
            with open(config_filename) as stream:
                cls._config = yaml_parser.load(stream) or dict()

        # Have we initialized the class member? If not initialize it empty
        if not hasattr(cls, '_config'):
            cls._config = dict()

        # The config member is either empty or initialized from the config file
        # Do we have a base currency setup? If not let's initialize it to USDT
        if 'base_currency' not in cls._config:
            cls._config['base_currency'] = "USDT"

        # Setting the default inventory_skew to 50%
        if 'inventory_skew' not in cls._config:
            cls._config['inventory_skew'] = 50

        # Setting the default weight of HBOT rewarded campaigns
        if 'hbot_weight' not in cls._config:
            cls._config['hbot_weight'] = 1

        # Have we loaded a hummingbot market definition
        if 'markets' not in cls._config:
            m_dict = dict(kucoin=dict(assets=set(), quotes=dict(), weights=dict(), hbot=set(), hbot_weights=dict(),
                                      inventory_skews=dict()),
                          gate_io=dict(assets=set(), quotes=dict(), weights=dict(), hbot=set(), hbot_weights=dict(),
                                       inventory_skews=dict()))
            m_dict['kucoin']['quotes'] = dict(
                USDT=["XCAD"],
                ETH=["NIM"],
                BTC=["VID"])
            m_dict['kucoin']['weights'] = {"XCAD-USDT": 1.5}
            m_dict['kucoin']['inventory_skews'] = {"XCAD-USDT": 30}
            m_dict['kucoin']['hbot'] = {"XCAD-USDT"}
            m_dict['kucoin']['hbot_weights'] = {"XCAD-USDT": 1.5}

            m_dict['gate_io']['quotes'] = dict(
                USDT=["XCAD"],
                ETH=["VSP"])
            m_dict['gate_io']['weights'] = {"XCAD-USDT": 1.5}
            m_dict['kucoin']['inventory_skews'] = {"XCAD-USDT": 30}
            m_dict['gate_io']['hbot'] = {"XCAD-USDT"}
            m_dict['gate_io']['hbot_weights'] = {"XCAD-USDT": 1.5}

            cls._config['markets'] = m_dict

    @classmethod
    def update_markets(cls, markets: Dict[str, Set[str]]) -> None:
        """
        Rearrange the markets passed as parameter into the appropriate format

        :param markets: markets definition, i.e. {'kucoin':{'ETH-USDT', 'BTC-USDT'}}
        """
        if markets and type(markets) == dict:
            cls._config['markets'] = dict()
            for exchange in markets:
                cls._config['markets'][exchange] = dict(assets=set(), quotes=dict(), weights=dict(), hbot=set(),
                                                        hbot_weights=dict(), inventory_skews=dict())
                for pairs in markets[exchange]:
                    base, quote = pairs.split('-')
                    cls._config['markets'][exchange]['assets'].add(base)
                    if quote not in cls._config['markets'][exchange]['quotes']:
                        cls._config['markets'][exchange]['quotes'][quote] = set()
                    cls._config['markets'][exchange]['quotes'][quote].add(base)

    @classmethod
    def initialize_markets(cls, config_filename: str) -> Dict[str, Set[str]]:
        """
        Initializes markets expected by hummingbot connectors initialization

        Returns:
            markets: expected by connectors initialization method

        :param config_filename: name of the configuration file to update
        """

        markets = dict()

        # Initializing the connectors
        for exchange_name in cls._config['markets']:
            markets[exchange_name] = set()
            cls._config['markets'][exchange_name]['assets'] = set()
            for quote in cls._config['markets'][exchange_name]['quotes']:
                # Adding connection to trades quotes other than base_currency -> base_currency
                if quote != cls._config['base_currency']:
                    markets[exchange_name].add(f"{quote}-{cls._config['base_currency']}")
                for asset in cls._config['markets'][exchange_name]['quotes'][quote]:
                    markets[exchange_name].add(f"{asset}-{quote}")
                    cls._config['markets'][exchange_name]['assets'].add(asset)

        cls._save_config_to_yml(config_filename)

        return markets

    @classmethod
    def _save_config_to_yml(cls, config_filename: str) -> None:
        """
        Saves the Lite Strategy script class configuration into the associated YAML file

        :param config_filename: name of the configuration file
        """
        if hasattr(cls, '_config'):
            with open(config_filename, "r") as stream:
                data = yaml_parser.load(stream) or {}
                for key in cls._config:
                    data[key] = cls._config.get(key)
                with open(config_filename, "w+") as outfile:
                    yaml_parser.dump(data, outfile)

    def get_assets_from_config(self, exchange: str) -> Set[str]:
        if exchange in self._config['markets']:
            return set(self._config['markets'][exchange]['assets'])
        else:
            return set()

    def get_campaigns_from_config(self, exchange: str) -> Dict:
        if exchange in self._config['markets']:
            return self._config['markets'][exchange]
        else:
            return dict()

    @property
    def base_currency(self) -> str:
        return self._config['base_currency']

    @property
    def hbot_weight(self) -> str:
        return self._config['hbot_weight']

    @property
    def inventory_skew(self) -> str:
        return self._config['inventory_skew']
