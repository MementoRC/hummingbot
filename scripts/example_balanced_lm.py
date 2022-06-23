import asyncio
import os
from typing import Dict, Set

from hummingbot.client.settings import CONF_FILE_PATH, CONF_PREFIX
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

# This implement the mechanics of importing the configuration from a YML file
# It is just convenient to have a separate file that all similar classes can import
from hummingbot.user.user_balances import UserBalances

from .funds_balancer import FundsBalancer
from .markets_yml_config import MarketsYmlConfig
from .trade_route_finder import TradeRouteFinder

lsb_logger = None


class ExampleBalancedLM(ScriptStrategyBase, MarketsYmlConfig):
    """
    Trying to get a better sense of balances and inventory in a common currency (USDT)
    """
    config_filename: str = CONF_FILE_PATH + CONF_PREFIX + os.path.split(__file__)[1].split('.')[0] + ".yml"
    valid_asset_route = {'BTC', 'USDT', 'ETH', 'USDC', 'DAI'}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self._balancing_trades = list()
        self._prices = dict()

        self._async_refresh = 45
        self._last_async_refresh_ts = 0

        self._trade_route_finder = None

        # Futures reference called with async on the main event loop (executed after the strategy tick())
        self._prices_fut = dict()
        self._balance_fut = dict()
        self._pause_fut = None

        self._data_ready = False

    @classmethod
    def initialize_from_yml(cls) -> Dict[str, Set[str]]:
        # Load the config or initialize with example configuration
        MarketsYmlConfig.load_from_yml(cls.config_filename)

        # Update the markets with local definition
        if hasattr(cls, 'markets'):
            MarketsYmlConfig.update_markets(cls.markets)

        # Return the markets for initialization of the connectors
        return MarketsYmlConfig.initialize_markets(cls.config_filename)

    def tick(self, timestamp: float):
        """
        Clock tick entry point, is run every second (on normal tick setting).
        Checks if all connectors are ready, if so the strategy is ready to trade.

        :param timestamp: current tick timestamp
        """
        if not self.ready_to_trade:
            self.ready_to_trade = all(ex.ready for ex in self.connectors.values())

            if not self.ready_to_trade:
                for con in [c for c in self.connectors.values() if not c.ready]:
                    self.logger().warning(f"{con.name} is not ready. Please wait...")
                return
        else:
            if self._last_async_refresh_ts < (self.current_timestamp - self._async_refresh):
                self._refresh_balances_prices_routes()
                self._last_async_refresh_ts = self.current_timestamp

            if self._data_ready:
                FundsBalancer.balancing_proposal(self, )
                self.on_tick()
            else:
                self.logger().warning("Strategy is not ready. Please wait...")
                return

    def _refresh_balances_prices_routes(self) -> None:
        """
        Calls async methods for all balance & price
        """
        self._data_ready = False
        loop = asyncio.get_event_loop()
        # We need balances to be updated and prices for both exchange (rather than use the oracle)
        # Submit to the main Event loop  - We get the result after a few ticks and wait till then
        if self._pause_fut is None:
            for exchange_name, connector in self.connectors.items():
                self._balance_fut[exchange_name] = asyncio.run_coroutine_threadsafe(
                    UserBalances.instance().update_exchange_balance(exchange_name), loop)
                if exchange_name == 'kucoin':
                    self._prices_fut['kucoin'] = asyncio.run_coroutine_threadsafe(RateOracle.get_kucoin_prices(), loop)
                elif exchange_name == 'gate_io':
                    self._prices_fut['gate_io'] = asyncio.run_coroutine_threadsafe(RateOracle.get_gate_io_prices(), loop)
            self._pause_fut = asyncio.run_coroutine_threadsafe(asyncio.sleep(0.5))

        for name, conn in self.connectors.items():
            if self._prices_fut[name].done():

                self._prices[name] = self._prices_fut[name].result()
                self._balancing_trades[name] = FundsBalancer.balancing_proposal(self, self._prices[name], conn)
                self._trade_route_finder[name] = TradeRouteFinder(self._prices[name], self.valid_asset_route)
                for trade in self._balancing_trades[name]:
                    trade['route'] = self._trade_route_finder[name].best_route(self._balancing_trades[name]['asset'],
                                                                               self._balancing_trades[name]['to'])

        if all([[self._balance_fut[c].done(), self._prices_fut[c].done()] for c in self.connectors.keys()]):
            self._data_ready = True
