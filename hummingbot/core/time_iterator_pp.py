from typing import Optional

from hummingbot.core.clock_pp import ClockPurePython

NaN = float("nan")


class TimeIteratorPurePython(object):
    def __init__(self):
        self._current_timestamp = NaN
        self._clock = None
        self._current_tick_size = NaN

    def start(self, clock: ClockPurePython):
        self._clock = clock
        self._current_timestamp = clock.current_timestamp
        self._current_tick_size = clock.tick_size

    def stop(self, clock: ClockPurePython):
        self._current_timestamp = NaN
        self._clock = None

    def create_ticks_till(self, clock: ClockPurePython):
        self._current_timestamp = NaN
        self._clock = None

    def tick(self, timestamp: float):
        self._current_timestamp = timestamp

    def next_tick(self):
        return self._clock.next_tick(self._current_timestamp, self.tick_size)

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
    def clock(self) -> Optional[ClockPurePython]:
        return self._clock

    def _set_current_timestamp(self, timestamp: float):
        """
        Method added to be used only for unit testing purposes
        """
        self._current_timestamp = timestamp
