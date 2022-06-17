import asyncio
import time
from typing import Dict, Set

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

from .ls_asyncio_event_loop_thread import LiteStrategyAsyncioEventLoopThread

lsb_logger = None


async def test_async_method(obj, wait_time):
    LiteStrategyAsyncCall.logger().info(f"{obj} - 1 execute for {wait_time / 4}s")
    await asyncio.sleep(wait_time / 4)
    LiteStrategyAsyncCall.logger().info(f"{obj} - 2 execute for {wait_time / 4}s")
    await asyncio.sleep(wait_time / 4)
    LiteStrategyAsyncCall.logger().info(f"{obj} - 3 execute for {wait_time / 4}s")
    await asyncio.sleep(wait_time / 4)
    LiteStrategyAsyncCall.logger().info(f"{obj} - 4 execute for {wait_time / 4}s")
    await asyncio.sleep(wait_time / 4)
    return "Async Result"


class LiteStrategyAsyncCall(ScriptStrategyBase):
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
        self.data_ready: bool = False
        self._prices = dict()

        self._last_async_refresh_ts = 0
        self._async_refresh = 10

        # Creating an event loop in a thread to call async methods
        self._thread = LiteStrategyAsyncioEventLoopThread()
        self._thread.start()

    # Testcase for calling an hummingbot async
    async def _async_calls(self) -> None:
        from hummingbot.core.management.diagnosis import active_tasks
        tasks_df = active_tasks().copy().reset_index()

        if tasks_df[tasks_df.func_name == 'get_kucoin_prices'].empty:
            self._prices["kucoin"] = await RateOracle.get_kucoin_prices()

    def on_tick(self):
        """
        An event which is called on every tick, a sub class implements this to define what operation the strategy needs
        to operate on a regular tick basis.
        """
        self.logger().info("Entering on_tick(): .")
        self.logger().info("Exiting on_tick(): .")
        pass

    def tick(self, timestamp: float):
        """
        Clock tick entry point, is run every second (on normal tick setting).
        Checks if all connectors are ready, if so the strategy is ready to trade.

        :param timestamp: current tick timestamp
        """
        self.logger().info("Entering tick()")
        if not self.ready_to_trade:
            self.ready_to_trade = all(ex.ready for ex in self.connectors.values())

            if not self.ready_to_trade:
                for con in [c for c in self.connectors.values() if not c.ready]:
                    self.logger().warning(f"{con.name} is not ready. Please wait...")
                return
        else:
            if self._last_async_refresh_ts < (self.current_timestamp - self._async_refresh):
                """
                Two ways to call async methods:
                    1 - On HB main event loop: Send the calls, they will be queued once returning
                    2 - Create an event loop in a thread, locally or persistent within the class
                        - Locally should be the safest
                        - Persistent seems to create some issues with HB event loop
                """

                self.logger().info("HB: Starting 3 events of different durations")
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(test_async_method("HB: 1", 1), loop=loop)
                asyncio.run_coroutine_threadsafe(test_async_method("HB: 2", 2), loop=loop)
                asyncio.run_coroutine_threadsafe(test_async_method("HB: 3", 3), loop=loop)
                self.logger().info("HB: 3 Events submitted")

                self.logger().info("Local: Starting event loop in a thread local to tick()")
                local_thread = LiteStrategyAsyncioEventLoopThread()
                local_thread.start()
                self.logger().info("Local: Starting 3 events of different durations")
                start_time = time.time()
                local_thread.run_and_wait_coro(test_async_method("Local: 1", 1), 4)
                local_thread.run_and_wait_coro(test_async_method("Local: 2", 2), 4)
                local_thread.run_and_wait_coro(test_async_method("Local: 3", 3), 4)
                self.logger().info(
                    f"Local: 3 Events started - Done in {time.time() - start_time} -  stopping the local thread")
                local_thread.stop()

                self.logger().info("Local CC: Starting event loop in a thread local to tick()")
                local_thread = LiteStrategyAsyncioEventLoopThread()
                local_thread.start()
                self.logger().info("Local CC: Starting 3 events via list")
                start_time = time.time()
                futures = local_thread.run_list_coro([test_async_method("Local CC: 1", 1),
                                                      test_async_method("Local CC: 2", 2),
                                                      test_async_method("Local CC: 3", 3)])
                [f.result() for f in futures]
                self.logger().info(
                    f"Local CC: 3 Events started - Done in {time.time() - start_time} - stopping the local thread")
                local_thread.stop()

                self.logger().info("self.: Starting event loop in a thread self. to tick()")
                self.logger().info("self.: Starting 3 events of different durations")
                start_time = time.time()
                self._thread.run_and_wait_coro(test_async_method("self.: 1", 1), 4)
                self._thread.run_and_wait_coro(test_async_method("self.: 2", 2), 4)
                self._thread.run_and_wait_coro(test_async_method("self.: 3", 3), 4)
                self.logger().info(
                    f"self.: 3 Events started - Done in {time.time() - start_time} -  stopping the self. thread")

                self.logger().info("self. CC: Starting event loop in a thread self. to tick()")
                self.logger().info("self. CC: Starting 3 events via list")
                start_time = time.time()
                futures = self._thread.run_list_coro([test_async_method("self. CC: 1", 1),
                                                      test_async_method("self. CC: 2", 2),
                                                      test_async_method("self. CC: 3", 3)])
                [f.result() for f in futures]
                self.logger().info(
                    f"self. CC: 3 Events started - Done in {time.time() - start_time} - stopping the self. thread")

                self._last_async_refresh_ts = self.current_timestamp

            if self.data_ready:
                self.on_tick()
                self.logger().info("Exiting tick()")
            else:
                self.logger().info("Exiting tick()")
                return

    def format_status(self) -> str:
        """
        Test the async execution from the status cmd
        """

        self.logger().info("status(): Async call")
        self._thread.run_coro(test_async_method(f"C - {self.current_timestamp}", 0))
        self.logger().info("status(): Done")

        return "Status Done"

    def stop(self, clock: Clock) -> str:
        """
        Stop the event loop
        """
        self.logger().info("Stopping the local event loop")
        self._thread.stop()
