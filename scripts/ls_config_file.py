import os
from typing import Dict, Set

from hummingbot.client.settings import CONF_FILE_PATH, CONF_PREFIX
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

# This implement the mechanics of importing the configuration from a YML file
# It is just convenient to have a separate file that all similar classes can import
from .ls_markets_init import LiteStrategyMarketsInit

lsb_logger = None


class LiteStrategyConfigFile(ScriptStrategyBase, LiteStrategyMarketsInit):
    """
    Trying to get a better sense of balances and inventory in a common currency (USDT)
    """
    markets: Dict[str, Set[str]]
    config_filename: str = CONF_FILE_PATH + CONF_PREFIX + os.path.split(__file__)[1].split('.')[0] + ".yml"

    @classmethod
    def initialize_from_yml(cls) -> Dict[str, Set[str]]:
        # Load the config or initialize with example configuration
        LiteStrategyMarketsInit.load_from_yml(cls.config_filename)

        # Update the markets with local definition
        if hasattr(cls, 'markets'):
            LiteStrategyMarketsInit.update_markets(cls.markets)

        # Return the markets for initialization of the connectors
        return LiteStrategyMarketsInit.initialize_markets(cls.config_filename)
