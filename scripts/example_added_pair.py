from typing import Dict, List, Set

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

from .asyncio_event_loop_thread import AsyncioEventLoopThread

lsb_logger = None


class ExampleAddedPair(ScriptStrategyBase):
    """
    Implementing call to async methods within Lite Strategy sync methods
    """
    markets = {"kucoin": {"ETH-BTC"}}

    last_ts = 0

    @classmethod
    def initialize_markets(cls, markets: Dict[str, Set[str]]) -> None:
        pass

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        #: Define markets to instruct Hummingbot to create connectors on the exchanges and markets you need
        super().__init__(connectors)
        self.added_pairs = False
        self.data_ready: bool = False
        self._prices = dict()

        self._last_async_refresh_ts = 0
        self._async_refresh = 15

        # Creating an event loop in a thread to call async methods
        self._thread = AsyncioEventLoopThread()
        self._thread.start()

    def on_tick(self):
        """
        An event which is called on every tick, a sub class implements this to define what operation the strategy needs
        to operate on a regular tick basis.
        """
        # self.logger().info("on_tick()")
        pairs = ['ETH-BTC', 'DOT-BTC']
        for pair in pairs:
            if pair in self.connectors['kucoin'].trading_pairs:
                self.logger().info(f"     Price check for {pair}: {self.connectors['kucoin'].get_price(pair, False)}")

    def tick(self, timestamp: float):
        """
        Clock tick entry point, is run every second (on normal tick setting).
        Checks if all connectors are ready, if so the strategy is ready to trade.

        :param timestamp: current tick timestamp
        """
        # self.logger().info("tick()")
        pairs = ['DOT-BTC']
        if self._last_async_refresh_ts == 0 or self._last_async_refresh_ts < (
                self.current_timestamp - self._async_refresh):
            self.logger().info(f"     Remove: {all([p in self.connectors['kucoin'].trading_pairs for p in pairs])}")
            if all([p in self.connectors['kucoin'].trading_pairs for p in pairs]):
                self.remove_pairs_subtle(pairs)
                self.added_pairs = True

            self.logger().info(f"     Add: {all([p not in self.connectors['kucoin'].trading_pairs for p in pairs])}")
            if all([p not in self.connectors['kucoin'].trading_pairs for p in pairs]):
                self.add_pairs_subtle(pairs)
                self.added_pairs = True

            self._last_async_refresh_ts = self.current_timestamp

        self.ready_to_trade = all(ex.ready for ex in self.connectors.values())
        if not self.ready_to_trade:
            if not self.ready_to_trade:
                for con in [c for c in self.connectors.values() if not c.ready]:
                    self.logger().warning(f"{con.name} is not ready. Please wait...")
                return
        else:
            self.on_tick()

    def add_pairs_brute_force(self, pairs):
        if self.ready_to_trade and all([p not in self.connectors['kucoin'].trading_pairs for p in pairs]):
            # self.added_pairs = True
            # Stopping the order book tracker to re-initialize it
            self.connectors['kucoin'].order_book_tracker.stop()
            # Adds the pairs
            self.connectors['kucoin'].trading_pairs.extend(pairs)
            # Restart the ordr book tracker - Could this create lost references to carefully prepared bot?
            self.connectors['kucoin'].order_book_tracker.start()

    def add_pairs_subtle(self, pairs: List[str]):
        if self.ready_to_trade and all([p not in self.connectors['kucoin'].trading_pairs for p in pairs]):
            self.connectors['kucoin'].order_book_tracker.add_orderbook_pairs(pairs)

    def remove_pairs_subtle(self, pairs: List[str]):
        if self.ready_to_trade and all([p in self.connectors['kucoin'].trading_pairs for p in pairs]):
            self.connectors['kucoin'].order_book_tracker.remove_orderbook_pairs(pairs)

    def format_status(self) -> str:
        """
        Test the async execution from the status cmd
        """
        return "Status Done"

    def stop(self, clock: Clock):
        """
        Stop the event loop
        """
        self.logger().info("Stopping AddedPair")
