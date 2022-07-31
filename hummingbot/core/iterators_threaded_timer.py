"""Provides a Thread based Timer class for Hummingbot TimeIterator instances.

This module encapsulates the interface provided by the internal special
attributes (co_*, im_*, tb_*, etc.) in a friendlier fashion.
It also provides some help for examining source code and class layout.

Here are some of the useful functions provided by this module:

    iterators() Provides the active list of iterators on the timer
    period_s() Provides the period of the timer in seconds
    add_iterator() Adds an iterator to the list
    remove_iterator() Removes an iterator from the list
    finished() Provides a boolean indicating if the finished event is set
    cancel() Terminates the timer
    restart() Restart a timer

    run() Internally executes the iterators repeatedly on its period
"""

# This module is in the public domain.  No warranties.

from __future__ import annotations

__author__ = ('Ka-Ping Yee <ping@lfw.org>',
              'Yury Selivanov <yselivanov@sprymix.com>')

import atexit
import logging
import threading
import time
import typing
from typing import List, Tuple

from hummingbot.core.clock_pp import ClockPurePython, ns_s, s_ns
from hummingbot.logger import HummingbotLogger

if typing.TYPE_CHECKING:
    from hummingbot.core.time_iterator_pp import TimeIteratorPurePython

s_logger = None


class IteratorsThreadedTimer(threading.Thread):
    @classmethod
    def logger(cls) -> HummingbotLogger:
        global s_logger
        if s_logger is None:
            s_logger = logging.getLogger(__name__)
        return s_logger

    def __init__(self, period: float, iterators: List["TimeIteratorPurePython"], timer_ns=time.perf_counter_ns, *args,
                 **kwargs):
        super(IteratorsThreadedTimer, self).__init__()
        self._args = args
        self._kwargs = kwargs
        self._finished = threading.Event()
        self._not_paused = threading.Event()
        self._not_paused.set()
        self._running = False
        self._was_paused = False

        self._period_s = period
        self._timer_ns = timer_ns
        self._s_iterators = [i for i in self._sort_by_priority(iterators) if i.tick_size == period]

        # Timing support
        self._task_duration_ns = 0

        # Registering cancel() for the 'CTRL+C' and Exception interrupt cleanup
        atexit.register(self.cancel)

    @property
    def period_s(self) -> float:
        """Returns the timer period in seconds.

        :returns: Timer period
        :rtype: float
        """
        return self._period_s

    @property
    def iterators(self) -> Tuple["TimeIteratorPurePython"]:
        """
        Returns the list of iterators executed by the timer.

        :returns: Tuple of the iterators
        :rtype: tuple[TimeIteratorPurePython]
        """
        return tuple(self._s_iterators)

    @property
    def finished(self) -> bool:
        """
        Returns the termination event that the loop is awaiting upon in read-only.

        :returns: Whether the finished event is set
        :rtype: bool
       """
        return self._finished.is_set()

    def run(self):
        """
        Awaits on its finished event with a timeout equal to the effective time to wait in
        order to maintain the execution of its iterators on the defined period of the timer
        A ValueError exception is raised by a subroutine if the execution of the iterators
        is longer than 105% of its period
        """
        self._running = True
        # Waits until the thread is terminated or the <period>> expires
        while not self._finished.wait(
                ns_s(period_ns := s_ns(self.period_s) - self._task_duration_ns)) and self._not_paused.wait():
            # Don't continue to execute if the cancel() is called
            print("\t.", end="")
            # Don't continue to execute if the cancel() is called
            if not self._finished.is_set() and self._was_paused:
                start_ns = self._timer_ns()
                # Execute the iterators
                self._run_iterators()
                # Update the duration
                self._task_duration_ns = self._timer_ns() - start_ns
                # Checking that the duration takes an appropriate time
                self._task_duration_ns = self._verify_duration(self._task_duration_ns, period_ns)
            self._was_paused = False

    def unpause(self):
        """Unpauses a paused timer"""
        if self._running:
            # This is to reset the timer
            self._was_paused = True
            self._task_duration_ns = time.time_ns() - ClockPurePython.get_current_tick_ns(s_ns(self._period_s))
            print(f"\tUnpausing in {self._period_s - ns_s(self._task_duration_ns)}s")
            self._not_paused.set()

    def pause(self):
        """Pauses the timer until unpause() is called."""
        print("\n\tPausing")
        self._not_paused.clear()

    def cancel(self):
        """Stops the timer."""
        print("\n\tCanceling the thread")
        self._running = False
        self._finished.set()

    def remove_iterator(self, iterator: "TimeIteratorPurePython"):
        """
        Removes an iterator from the list of iterators.

        :param TimeIteratorPurePython iterator: Iterator to add to the list of iterators executed
        """
        self._s_iterators.remove(iterator)

    def add_iterator(self, iterator: "TimeIteratorPurePython"):
        """
        Adds an iterator to the list of iterators and sort
        Verifies that the iterator is not present and that its tick_size is compatible.

        :param TimeIteratorPurePython iterator: Iterator to add to the list of iterators executed
        """
        if iterator.tick_size == self._period_s and iterator not in self._s_iterators:
            self._s_iterators = self._sort_by_priority([iterator] + self._s_iterators)

    @staticmethod
    def _sort_by_priority(iterators: List["TimeIteratorPurePython"]) -> List["TimeIteratorPurePython"]:
        """
        Returns the list of iterators sorted by Iterator priority (uses 0 as default priority).

        :param list[TimeIteratorPurePython] iterators: List of the iterators to sort
        :returns: Sorted list of the iterators
        :rtype: list[TimeIteratorPurePython]
        """
        return sorted(iterators, key=lambda it: it.priority if hasattr(it, "priority") else 0, reverse=True)

    def _run_iterators(self):
        """Executes the tick() method of the iterator (only if there is such method)."""
        [it.tick(ClockPurePython.get_current_tick_s(self._period_s)) for it in self._s_iterators if
         hasattr(it, 'tick') and callable(it.tick)]

    def _verify_duration(self, duration_ns: int, period_ns: int) -> int:
        """
        Warns if the duration of the iterators is larger than 95% the period of the timer.
        Warns and skip the next execution if larger than 99% but smaller than 105% the period of the timer.
        Raises and ValueError exception if larger than 105% the period of the timer. This is to allow one-off delay

        :param int duration_ns: Duration to be verified
        :param int period_ns: Effective period the compare the duration to
        :returns: The duration, updated for period skipping case
        :rtype: int
        :raises ValueError: if the duration is larger than 105% of the period
        """
        if duration_ns < (period_ns * 95) // 100:
            pass
        elif (period_ns * 95) // 100 <= duration_ns < (period_ns * 99) // 100:
            self.logger().warning(f"The iterators on {self.period_s}s ticks are taking:\n"
                                  f"\t{ns_s(duration_ns)}s ({duration_ns // period_ns}%) to execute")
        elif (period_ns * 99) // 100 <= duration_ns < (period_ns * 105) // 100:
            duration_ns = duration_ns - s_ns(self.period_s)
            self.logger().warning(f"The iterators on {self.period_s}s ticks are taking too long:\n"
                                  f"\tSkipping one tick, executing in {self.period_s - ns_s(duration_ns)}s")
        else:
            self.logger().error(f"The iterators on {self.period_s}s ticks are taking too long:\n"
                                f"\tStopping this iterator and raising an exception")
            raise ValueError(f"Iterators setting for tick_size: {self.period_s} invalid")
        return duration_ns
