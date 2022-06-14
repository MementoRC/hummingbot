from os.path import isfile
from typing import Dict, List, Set, Union

import ruamel.yaml

HMarkets = Dict[str, Dict[str, Union[Dict[str, List[str]], List[str]]]]
LSConfig = Dict[str, Union[HMarkets, List[str], Set[str]]]


yaml_parser = ruamel.yaml.YAML()


class LiteStrategyMarketInit(object):
    """
    Testing a market initialization method for Script Strategies
    Initialize the list of markets prior the market initialization phase in Hummingbot
    """
    HMarkets = Dict[str, Union[Dict[str, List[str]], Set[str]]]
    Config = Dict[str, Union[str, int, HMarkets, Set[str]]]

    _config: Config

    @classmethod
    def initialize_markets(cls, markets: Dict[str, Set[str]], config_filename: str):
        """
        Initializes the class variables prior to instantiation
        This class should be imported from importlib.import_module function
            it requires hummingbot initialization prior to being instantiated
            thus the markets need to be defined as a class member 'markets'
            however, it is also convenient to be able to modify markets from a
            configuration file

        :param markets: markets class member to be modified (prior to class instantiation)
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

        # Have we loaded a hummingbot market definition and is it valid
        if 'markets' not in cls._config:
            m_dict = dict(kucoin=dict(assets=set(), quotes=dict(), weights={}, hbot=set(), hbot_weight=1),
                          gate_io=dict(assets=set(), quotes=dict(), weights={}, hbot=set(), hbot_weight=1))
            m_dict['kucoin']['quotes'] = dict(
                USDT=["ALGO", "AVAX", "DAO", "FEAR", "FRONT", "HAI", "HOTCROSS", "NIM", "ODDZ", "ONE", "SHR", "VID",
                      "XCAD"], ETH=["NIM"], BTC=["ALGO", "FRONT", "NIM", "SHR", "VID"])
            m_dict['kucoin']['weights'] = {"ALGO-USDT": 1.5}
            m_dict['kucoin']['inventory_skews'] = {"ALGO-USDT": 30}
            m_dict['kucoin']['hbot'] = {"ALGO-BTC", "FRONT-BTC", "SHR-BTC", "ALGO-USDT", "AVAX-USDT", "DAO-USDT",
                                        "FEAR-USDT", "FRONT-USDT", "HAI-USDT", "ONE-USDT", "SHR-USDT"}
            m_dict['kucoin']['hbot_weights'] = {"ALGO-USDT": 1.5}

            m_dict['gate_io']['quotes'] = dict(
                USDT=["AVAX", "FEAR", "FIRO", "HAI", "HMT", "LIME", "ODDZ", "VSP", "XCAD"], ETH=["HMT", "VSP"])
            m_dict['gate_io']['weights'] = {"AVAX-USDT": 1.5}
            m_dict['kucoin']['inventory_skews'] = {"AVAX-USDT": 30}
            m_dict['gate_io']['hbot'] = {"AVAX-USDT", "FEAR-USDT", "FIRO-USDT", "HAI-USDT"}
            m_dict['gate_io']['hbot_weights'] = {"AVAX-USDT": 1.5}

            cls._config['markets'] = m_dict

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

        cls.save_markets_to_yml(config_filename)

    @classmethod
    def save_markets_to_yml(cls, config_filename: str):
        """
        Saves the Lite Strategy script class configuration into the associated YAML file

        :param config_filename: name of the configuration file
        """
        with open(config_filename, "r") as stream:
            data = yaml_parser.load(stream) or {}
            for key in cls._config:
                data[key] = cls._config.get(key)
            with open(config_filename, "w+") as outfile:
                yaml_parser.dump(data, outfile)
