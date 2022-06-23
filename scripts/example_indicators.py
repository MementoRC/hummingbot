from os.path import dirname

import pandas as pd

from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from unofficial_addendum.plot_indicators import AddIndicators
from unofficial_addendum.utils import Plot_OHCL


class ExampleIndicators(ScriptStrategyBase):
    """
    Testing how to access Indicators
    """
    markets = {"kucoin": {"ETH-BTC"}}

    def format_status(self) -> str:
        """
        Returns status of the current strategy on user balances and current active orders. This function is called
        when status command is issued. Override this function to create custom status display output.
        """
        df = pd.read_csv(dirname(__file__) + '/../unofficial_addendum/pricedata.csv')
        df = df.sort_values('Date')
        df = AddIndicators(df)

        test_df = df[-400:]

        Plot_OHCL(test_df)
        if not self.ready_to_trade:
            return "Market connectors are not ready."
