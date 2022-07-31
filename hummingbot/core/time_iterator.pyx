# distutils: language=c++
from typing import Optional

from hummingbot.core.clock import Clock

NaN = float("nan")


cdef class TimeIterator(PubSub):
    def __init__(self):
        self._current_timestamp = NaN
        self._clock = None
        self._current_tick_size = NaN

    cdef c_start(self, Clock clock, double timestamp):
        self._clock = clock
        self._current_timestamp = timestamp
        self._current_tick_size = clock.tick_size

    cdef c_stop(self, Clock clock):
        self._current_timestamp = NaN
        self._clock = None

    cdef c_tick(self, double timestamp):
        self._current_timestamp = timestamp

    def tick(self, timestamp: float):
        self.c_tick(timestamp)

    @property
    def tick_size(self) -> float:
        return self._current_tick_size

    @tick_size.setter
    def tick_size(self, tick_size: float):
        self._current_tick_size = tick_size

    @property
    def current_timestamp(self) -> float:
        return self._current_timestamp

    @property
    def clock(self) -> Optional[Clock]:
        return self._clock

    def start(self, clock: Clock):
        self.c_start(clock, clock.current_timestamp)

    def stop(self, clock: Clock):
        self.c_stop(clock)

    def _set_current_timestamp(self, timestamp: float):
        """
        Method added to be used only for unit testing purposes
        """
        self._current_timestamp = timestamp
