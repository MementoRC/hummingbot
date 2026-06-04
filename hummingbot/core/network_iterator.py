import asyncio
import logging
from enum import Enum
from typing import Optional

from async_utils.core import safe_ensure_future

from hummingbot.core.time_iterator import TimeIterator
from hummingbot.logger import HummingbotLogger

NaN = float("nan")
ni_logger = None


class NetworkStatus(Enum):
    STOPPED = 0
    NOT_CONNECTED = 1
    CONNECTED = 2


class NetworkIterator(TimeIterator):
    @classmethod
    def logger(cls) -> HummingbotLogger:
        global ni_logger
        if ni_logger is None:
            ni_logger = logging.getLogger(__name__)
        return ni_logger

    def __new__(cls, *args, **kwargs):
        # Ensure network state fields exist before __init__ completes, as connector
        # subclasses sometimes skip super().__init__() chains.
        instance = super().__new__(cls)
        instance._network_status = NetworkStatus.STOPPED
        instance._last_connected_timestamp = float("nan")
        instance._check_network_interval = 10.0
        instance._check_network_timeout = 5.0
        instance._network_error_wait_time = 60.0
        instance._check_network_task = None
        return instance

    def __init__(self):
        super().__init__()
        self._network_status = NetworkStatus.STOPPED
        self._last_connected_timestamp = NaN
        self._check_network_interval = 10.0
        self._check_network_timeout = 5.0
        self._network_error_wait_time = 60.0
        self._check_network_task = None

    @property
    def network_status(self) -> NetworkStatus:
        return self._network_status

    @property
    def last_connected_timestamp(self) -> float:
        return self._last_connected_timestamp

    @last_connected_timestamp.setter
    def last_connected_timestamp(self, value: float) -> None:
        self._last_connected_timestamp = value

    @property
    def check_network_task(self) -> Optional[asyncio.Task]:
        return self._check_network_task

    @property
    def check_network_interval(self) -> float:
        return self._check_network_interval

    @check_network_interval.setter
    def check_network_interval(self, interval: float) -> None:
        self._check_network_interval = interval

    @property
    def network_error_wait_time(self) -> float:
        return self._network_error_wait_time

    @network_error_wait_time.setter
    def network_error_wait_time(self, wait_time: float) -> None:
        self._network_error_wait_time = wait_time

    @property
    def check_network_timeout(self) -> float:
        return self._check_network_timeout

    @check_network_timeout.setter
    def check_network_timeout(self, timeout: float) -> None:
        self._check_network_timeout = timeout

    async def start_network(self) -> None:
        pass

    async def stop_network(self) -> None:
        pass

    async def check_network(self) -> NetworkStatus:
        self.logger().warning("check_network() has not been implemented!")
        return NetworkStatus.NOT_CONNECTED

    async def _check_network_loop(self) -> None:
        while True:
            last_status = self._network_status
            try:
                new_status = await asyncio.wait_for(self.check_network(), timeout=self._check_network_timeout)
                if new_status != last_status:
                    self._network_status = new_status
                    if self._network_status is NetworkStatus.CONNECTED:
                        self.logger().info(f"Network status has changed to {new_status}. Starting networking...")
                        await self.start_network()
                    else:
                        self.logger().info(f"Network status has changed to {new_status}. Stopping networking...")
                        await self.stop_network()
                await asyncio.sleep(self._check_network_interval)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                self.logger().debug("Check network call has timed out. Network status is not connected.")
                self._network_status = NetworkStatus.NOT_CONNECTED
                await asyncio.sleep(self._check_network_interval)
            except Exception as e:
                self.logger().error(
                    f"Unexpected error while checking for network status: {e}",
                    exc_info=True,
                )
                self._network_status = NetworkStatus.NOT_CONNECTED
                await asyncio.sleep(self._network_error_wait_time)

    def tick(self, timestamp: float) -> None:
        super().tick(timestamp)

    def start(self, clock, timestamp: float) -> None:
        super().start(clock, timestamp)
        self._check_network_task = safe_ensure_future(self._check_network_loop())
        self._network_status = NetworkStatus.NOT_CONNECTED

    def stop(self, clock) -> None:
        super().stop(clock)
        if self._check_network_task is not None:
            self._check_network_task.cancel()
            self._check_network_task = None
        self._network_status = NetworkStatus.STOPPED
        safe_ensure_future(self.stop_network())
