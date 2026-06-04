from typing import Optional

from hummingbot.core.pubsub import PubSub

NaN = float("nan")


class TimeIterator(PubSub):
    def __new__(cls, *args, **kwargs):
        # Mirrors Cython C-level allocation. Cython Clock dispatches c_tick
        # before any Python __init__ chain may have completed for skipping
        # subclasses; _current_timestamp must exist as NaN (sentinel for
        # "unset") and _clock as None.
        instance = super().__new__(cls)
        instance._current_timestamp = float("nan")
        instance._clock = None
        return instance

    def __init__(self):
        self._current_timestamp: float = NaN
        self._clock = None

    def c_start(self, clock, timestamp: float) -> None:
        self._clock = clock
        self._current_timestamp = timestamp

    def c_stop(self, clock) -> None:
        self._current_timestamp = NaN
        self._clock = None

    def c_tick(self, timestamp: float) -> None:
        self._current_timestamp = timestamp

    def tick(self, timestamp: float) -> None:
        self.c_tick(timestamp)

    @property
    def current_timestamp(self) -> float:
        return self._current_timestamp

    @property
    def clock(self) -> Optional[object]:
        return self._clock

    def start(self, clock) -> None:
        self.c_start(clock, clock.current_timestamp)

    def stop(self, clock) -> None:
        self.c_stop(clock)

    def _set_current_timestamp(self, timestamp: float) -> None:
        """
        Method added to be used only for unit testing purposes
        """
        self._current_timestamp = float(timestamp)
