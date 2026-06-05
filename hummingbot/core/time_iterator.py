from typing import Optional

from hummingbot.core.pubsub import PubSub

NaN = float("nan")


class TimeIterator(PubSub):
    def __new__(cls, *args, **kwargs):
        # Ensure _current_timestamp and _clock exist before __init__ completes,
        # as subclasses may skip the super().__init__() chain.
        instance = super().__new__(cls)
        instance._current_timestamp = 0.0
        instance._clock = None
        return instance

    def __init__(self):
        self._current_timestamp: float = 0.0
        self._clock = None

    def tick(self, timestamp: float) -> None:
        self._current_timestamp = timestamp

    @property
    def current_timestamp(self) -> float:
        return self._current_timestamp

    @property
    def clock(self) -> Optional[object]:
        return self._clock

    def start(self, clock, timestamp: float | None = None) -> None:
        self._clock = clock
        if timestamp is None:
            timestamp = clock.current_timestamp
        self._current_timestamp = timestamp

    def stop(self, clock) -> None:
        self._current_timestamp = NaN
        self._clock = None

    def _set_current_timestamp(self, timestamp: float) -> None:
        """
        Method added to be used only for unit testing purposes
        """
        self._current_timestamp = float(timestamp)
