from __future__ import annotations

import asyncio
import functools
import itertools
import logging
import time
import typing
from collections import OrderedDict, defaultdict
from decimal import Decimal
from functools import reduce
from math import gcd
from typing import Dict, List, Set, Tuple, Union

from hummingbot.core.clock_mode import ClockMode
from hummingbot.core.iterators_threaded_timer import IteratorsThreadedTimer
from hummingbot.logger import HummingbotLogger

if typing.TYPE_CHECKING:
    from hummingbot.core.time_iterator_pp import TimeIteratorPurePython

s_logger: HummingbotLogger = None


def ns_s(timestamp: Union[int, str]) -> float:
    return float(Decimal(timestamp) * Decimal("1e-9"))


def s_ns(timestamp: Union[float, str]) -> int:
    return int(Decimal(timestamp) * Decimal("1e9"))


async def try_except_async(function):
    async def call_tick(*args, **kwargs):
        returned_value = None
        try:
            returned_value = await function(*args, **kwargs)
        except StopIteration:
            ClockPurePython.logger().error("Stop iteration triggered in real time mode. This is not expected.")
            return
        except Exception:
            ClockPurePython.logger().error("Unexpected error running clock tick.", exc_info=True)
        finally:
            return returned_value

    return call_tick


def try_except(function):
    def call_tick(*args, **kwargs):
        returned_value = None
        try:
            returned_value = function(*args, **kwargs)
        except StopIteration:
            ClockPurePython.logger().error("Stop iteration triggered in real time mode. This is not expected.")
            return
        except Exception:
            ClockPurePython.logger().error("Unexpected error running clock tick.", exc_info=True)
        finally:
            return returned_value

    return call_tick


@try_except
def in_executor(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        loop = asyncio.get_event_loop()
        func = functools.partial(function, *args, **kwargs)
        return loop.run_in_executor(executor=None, func=func)

    return wrapped


class ClockPurePython:
    @classmethod
    def logger(cls) -> HummingbotLogger:
        global s_logger
        if s_logger is None:
            s_logger = logging.getLogger(__name__)
        return s_logger

    TimeType = Union[str, Decimal, float, int]
    smallest_tick_size: TimeType = 10000000  # in nanoseconds

    def __init__(self, clock_mode: ClockMode, tick_size: TimeType = 1.0, start_time: TimeType = 0.0,
                 end_time: TimeType = 0.0, is_nanoseconds=False):
        """
        :param clock_mode: either real time mode or back testing mode
        :param tick_size: time interval of each tick
        :param start_time: (back testing mode only) start of simulation in UNIX timestamp
        :param end_time: (back testing mode only) end of simulation in UNIX timestamp. NaN to simulate to end of data.
        """
        self._clock_mode = clock_mode
        self._child_iterators = []
        self._current_context = None
        self._ticks_cycle = None
        self._started = False

        if is_nanoseconds:
            self._tick_size = int(max(tick_size, ClockPurePython.smallest_tick_size))
            self._start_time = int(start_time)
            self._end_time = int(end_time)
        else:
            self._tick_size = int(max(s_ns(tick_size), ClockPurePython.smallest_tick_size))
            self._start_time = s_ns(start_time)
            self._end_time = s_ns(end_time)

        self._current_tick = self._start_time if clock_mode is ClockMode.BACKTEST else self._tick_formula(
            time.time_ns(), self._tick_size)
        self._next_tick = self._current_tick + self._tick_size

    @property
    def clock_mode(self) -> ClockMode:
        return self._clock_mode

    @property
    def start_time(self) -> float:
        return ns_s(self._start_time)

    @property
    def child_iterators(self) -> List["TimeIteratorPurePython"]:
        return self._child_iterators

    @property
    def current_timestamp_ns(self) -> int:
        return self._current_tick

    @property
    def current_timestamp(self) -> float:
        return ns_s(self._current_tick)

    @staticmethod
    def get_current_tick_s(tick_s: float) -> float:
        return ns_s(ClockPurePython._tick_formula(time.time_ns(), s_ns(tick_s)))

    @staticmethod
    def get_current_tick_ns(tick_ns: int) -> int:
        return ClockPurePython._tick_formula(time.time_ns(), tick_ns)

    def __enter__(self):
        if self._current_context is not None:
            self.logger().error("Clock context is not re-entrant.")
            raise EnvironmentError("Clock context is not re-entrant.")
        self._current_context = self._child_iterators.copy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._current_context is not None:
            for iterator in self._current_context:
                iterator.stop(self)
                self._ticks_cycle['timer_for_iterator'][iterator].remove_iterator(iterator)
            [t.cancel() for t in self._ticks_cycle['iterators_timers']]
            self._current_context = None

    def add_iterator(self, iterator: "TimeIteratorPurePython"):
        if self._current_context is not None:
            self._current_context.append(iterator)
            if self._started:
                if iterator.tick_size in (i.tick_size for i in self._sorted_context):
                    [t.add_iterator(iterator) for t in self._ticks_cycle['iterators_timers']]
                else:
                    self.logger().error("New iterator tick_size incompatible with current clock.")
                    raise EnvironmentError("New iterator tick_size incompatible with current clock.")
                iterator.start(self, self._current_tick)
        self._child_iterators.append(iterator)

    def remove_iterator(self, iterator: "TimeIteratorPurePython"):
        if self._current_context is not None and iterator in self._current_context:
            iterator.stop(self)
            # Remove from its timer
            self._ticks_cycle['timer_for_iterator'][iterator].remove_iterator(iterator)
            self._current_context.remove(iterator)
        self._child_iterators.remove(iterator)

    def _start_clocking_in_context(self):
        if self._current_context is None:
            self.logger().error("Context is not initialized")
            raise EnvironmentError("Context is not initialized")
        if self._sorted_context is None:
            self.logger().error("Context is not configured for Timers")
            raise EnvironmentError("Context is not configured for Timers")
        # Start the clock to pass to the iterators
        self._set_current_tick()
        # Start
        if not self._started:
            # Start the iterators in context
            [ci.start(self) for ci in self._current_context]
            # Start the timers in new threads
            [t.start() for t in self._ticks_cycle['iterators_timers']]
            # Timers will only execute after the first period, running all iterators once
            [it.tick() for it in self._ticks_cycle['iterators_cycle'][0]]
            self._started = True
        else:
            self.logger().error("Attempt to restart a clock: Not implemented")
            raise EnvironmentError("Attempt to restart a clock: Not implemented")

    def _sort_context(self):
        if self._current_context is None:
            self.logger().error("Context is not initialized")
            raise EnvironmentError("Context is not initialized")

        if len(self._current_context) == 0:
            self.logger().error("Context is empty")
            raise EnvironmentError("Context is empty")
        self._sorted_context = sorted(list(set([i for i in self._current_context])),
                                      key=lambda i: (i.tick_size, i.priority if hasattr(i, 'priority') else 0))

    def _construct_iterators_timetable(self) -> None:
        """
        Creates the Clock tick intervals:
          The clock starts at a given time
          The intervals are added one after another to calculate the next tick
        Creates the dictionary of set of iterators to run at each Clock ticks
          Tick intervals and tick time are int, thus easy to sort
          dict[0]: at Clock start, set of all the iterators
          dict[1st clock tick]: set of iterators with the smallest tick_size
          dict[2nd clock tick]: set of iterators with the smallest tick_size or (non-exclusive)
                                set of iterators with the 2nd smallest tick_size
          ...
        """
        if self._current_context is None:
            self.logger().error("Context is not initialized")
            raise EnvironmentError("Context is not initialized")

        if len(self._current_context) == 0:
            self.logger().error("Context is empty")
            raise EnvironmentError("Context is empty")

        dict_iterators: Dict[int, Set["TimeIteratorPurePython"]] = defaultdict(set)

        # List of unique tick sizes, ordered that constitute the base of the Clock ticks
        # Those are the base intervals on which each iterator clocks
        self._sort_context()
        base_intervals = [i.tick_size for i in self._sorted_context]
        # Calculate the lowest common multiple of the primary tick_sizes, i.e base_intervals
        last_tick_time = reduce(lambda a, b: a * b // gcd(a, b), base_intervals)
        # All iterators start with the Clock (start time is not know at initialization)
        dict_iterators[0] |= set(self._current_context)
        iterators_timers = []
        timer_for_iterator = {}

        # The dict automatically removes duplicated time and simply add the needed iterators
        #   At the end the dict is replaced by a OrderedDict
        for base_interval in base_intervals:
            # Initialize the tick_time to t=0 + base_interval
            tick_time = base_interval
            # Collect the list of iterators with tick_size == tick_interval
            interval_iterators = {ci for ci in self._current_context if ci.tick_size == base_interval}
            timer: IteratorsThreadedTimer = IteratorsThreadedTimer(ns_s(base_interval), list(interval_iterators))
            [timer_for_iterator.update({i: timer}) for i in list(interval_iterators)]
            iterators_timers.append(timer)
            iterators_timers[-1].name = iterators_timers[-1].name + f":{base_interval}s"
            # Append the list of all the tick times generated by repeating this interval or period
            # till the last tick time
            while tick_time <= last_tick_time:
                # Create/Append the timetable for the iterator with a tick size == tick_interval
                if tick_time in dict_iterators:
                    dict_iterators[tick_time].update(interval_iterators)
                else:
                    dict_iterators[tick_time] |= interval_iterators
                tick_time = tick_time + base_interval
        # Register the endless cycle iterator in the instance
        it_cycle = OrderedDict({k: dict_iterators[k] for k in sorted(dict_iterators)})
        self._ticks_cycle = dict()
        self._ticks_cycle['iterators_cycle'] = it_cycle
        self._ticks_cycle['iterators_timers'] = iterators_timers
        self._ticks_cycle['timer_for_iterator'] = timer_for_iterator
        # List of the succession of intervals between the Clock ticks
        self._ticks_cycle['tick_intervals'] = [j - i for i, j in
                                               zip(list(it_cycle.keys())[:-1], list(it_cycle.keys())[1:])]
        self._ticks_cycle['intervals_cycle_iterator'] = itertools.cycle(self._ticks_cycle['tick_intervals'])

    @staticmethod
    def _tick_formula(time_in_ns: int, tick_size_in_ns: int, is_next: bool = False) -> int:
        d = (time_in_ns // tick_size_in_ns + int(is_next)) * tick_size_in_ns
        return d

    def _set_current_tick(self) -> Tuple[int, int]:
        c_tick = n_tick = time.time_ns()
        for i in self._sorted_context:
            c_tick = min(c_tick, ClockPurePython._tick_formula(time.time_ns(), i.tick_size, False))
            n_tick = max(n_tick, ClockPurePython._tick_formula(time.time_ns(), i.tick_size, True))
        self._current_tick, self._next_tick = c_tick, n_tick
        return self._current_tick, self._next_tick

    async def run(self):
        await self.run_till_in_timers(float("nan"))

    @staticmethod
    def _busy_wait_ns(delay: int,
                      accuracy: int = 10000,
                      timer_ns: callable = time.perf_counter_ns) -> Dict[str, int]:
        """
        Busy Wait timer. The minimum delta measurable on my machine it 1
        """
        start: int = timer_ns()

        if delay < accuracy // 2:
            wait = timer_ns() - start
            return {"current_time": start,
                    "busy_count": 0,
                    "busy_wait": 0,
                    "busy_residual": 0,
                    "effective_wait": wait,
                    "residual": 0,
                    "changes": 0,
                    }

        count = 0
        end: int = start + delay - accuracy // 2
        while end - timer_ns() >= 0:
            for i in range(250):
                x = 10000000 ** 2 // (10000000 + 1)
                x = x + 1
            count = count + 1
        current_time = timer_ns()
        wait = current_time - start
        residual = wait - delay
        return {"current_time": current_time,
                "busy_wait": wait,
                "busy_residual": residual,
                "effective_wait": wait,
                "residual": residual,
                "busy_count": count,
                "changes": [],
                }

    @staticmethod
    def _time_sleep_ns(delay: int, accuracy: int = 500000, timer_ns: callable = time.perf_counter_ns,
                       busy_wait: bool = True) -> Dict[str, int]:
        start: int = timer_ns()
        timing_data: Dict[str, int] = dict(sleep0_wait=0,
                                           sleep1_wait=0,
                                           sleep_count=0,
                                           )
        if delay < accuracy:
            if not busy_wait:
                return {"current_time": timer_ns(),
                        "busy_wait": 0,
                        "busy_residual": 0,
                        "busy_count": 0,
                        "effective_wait": timer_ns() - start,
                        "residual": delay - (timer_ns() - start),
                        "sleep0_wait": 0,
                        "sleep1_wait": 0,
                        "sleep_count": 0,
                        }
            wait: Dict[str, int] = ClockPurePython._busy_wait_ns(delay, timer_ns=timer_ns)
            wait.update(timing_data)
            return wait

        # The sleep(0) is equivalent "busy_wait" in the sense that it relinquishes control
        elif delay < 800000:
            min_sleep_time: float = 0
            sleep_offset: int = 2000
        # The minimum sleep time on my machine was 80-160us
        else:
            min_sleep_time: float = max(1e-9, ns_s(delay) / 50)
            sleep_offset: int = 50000

        end: int = start + delay - sleep_offset
        count = 0
        monitor = timer_ns()
        while end - monitor >= 0:
            time.sleep(min_sleep_time)
            count = count + 1
            monitor = timer_ns()
            if start + delay - monitor < sleep_offset:
                timing_data['sleep_count'] = count
                if min_sleep_time == 0:
                    timing_data['sleep0_wait'] = monitor - start
                else:
                    timing_data['sleep1_wait'] = monitor - start
                if busy_wait:
                    end: int = start + delay
                    wait = ClockPurePython._busy_wait_ns(end - timer_ns())
                    wait['effective_wait'] = wait['current_time'] - start
                    wait['residual'] = wait['effective_wait'] - delay
                    wait.update(timing_data)
                    return wait
                wait = {"current_time": timer_ns(),
                        "busy_wait": 0,
                        "busy_residual": 0,
                        "busy_count": 0,
                        "effective_wait": monitor - start,
                        "residual": delay - (monitor - start),
                        }
                wait.update(timing_data)
                return wait
        timing_data['sleep_count'] = count
        wait = {"current_time": timer_ns(),
                "busy_wait": 0,
                "busy_residual": 0,
                "busy_count": 0,
                "effective_wait": monitor - start,
                "residual": delay - (monitor - start),
                }
        wait.update(timing_data)
        return wait

    @staticmethod
    async def _async_sleep_ns(delay: int, async_threshold: int = 50000000, async_accuracy: int = 10000000,
                              timer_ns: callable = time.perf_counter_ns, block_sleep: bool = True,
                              throttle: bool = False) -> Dict[str, int]:
        start = timer_ns()
        timing_data: Dict[str, int] = dict(async_wait=0,
                                           sync_wait=0,
                                           current_time=0,
                                           busy_wait=0,
                                           busy_residual=0,
                                           busy_count=0,
                                           sleep_count=0,
                                           sleep0_wait=0,
                                           sleep1_wait=0,
                                           effective_wait=0,
                                           residual=0,
                                           )
        if delay < async_threshold:
            ClockPurePython.logger().warning(f"Asyncio.sleep({ns_s(delay)}s/{delay}ns) requested too small. {50}ms "
                                             f"accuracy. Using blocking time.sleep().")
            if block_sleep:
                timing_data.update(ClockPurePython._time_sleep_ns(delay, timer_ns=timer_ns, busy_wait=True))
            else:
                monitor = timer_ns()
                timing_data.update({"current_time": monitor,
                                    "effective_wait": monitor - start,
                                    "residual": delay - (monitor - start),
                                    })
        else:
            start: int = timer_ns()
            if not throttle:
                await asyncio.sleep(ns_s(delay - async_accuracy // 2))
            else:
                end: int = start + delay
                while (remain := end - timer_ns()) >= 0:
                    wait = min(remain // 2, 1000000)
                    await asyncio.sleep(ns_s(wait))
            async_wait = timer_ns() - start
            if block_sleep:
                timing_data.update(
                    ClockPurePython._time_sleep_ns(async_wait - delay, timer_ns=timer_ns, busy_wait=True))
            timing_data['sync_wait'] = timing_data['effective_wait']
            timing_data['async_wait'] = async_wait
            timing_data['effective_wait'] = timer_ns() - start
            timing_data['residual'] = timing_data['effective_wait'] - delay
        return timing_data

    def _profile(self, func):
        def call_f(*args, **kwargs):
            self.pr.enable()
            returned_value = func(*args, **kwargs)
            self.pr.disable()
            return returned_value

        return call_f

    async def run_till_in_executor(self, timestamp: float):
        if self._current_context is None:
            self.logger().error("run() and run_til() can only be used within the context of a `with...` statement.")
            raise EnvironmentError("run() and run_til() can only be used within the context of a `with...` statement.")

        self._construct_iterators_timetable()

        slew: Dict[str, Union[int, List[Dict[str, int]]]] = {'asyncio.sleep': 0,
                                                             'total_cycle': 0,
                                                             'iterators': [],
                                                             'iterators_cycle': 0,
                                                             'previous_loop_adjust': 0,
                                                             }
        #
        relative_tick_time = 0
        self._start_clocking_in_context()
        try:
            while True:
                # Exit condition
                if self._current_tick > timestamp:
                    return
                start_timer = time.perf_counter_ns()
                for iterator in self._ticks_cycle['iterators_cycle'][relative_tick_time]:
                    start_exec_timer = time.perf_counter_ns()
                    await in_executor(iterator.tick(self._current_tick))
                    slew['iterators'].append({f"{iterator.display_name}": time.perf_counter_ns() - start_exec_timer})
                slew['iterators_cycle'] = time.perf_counter_ns() - start_timer

                # Key iteration procedure
                interval_to_next_tick = next(self._ticks_cycle['tick_intervals'])
                wait_time = interval_to_next_tick - slew['iterators_cycle'] - slew['previous_loop_adjust']

                start_timer = time.perf_counter_ns()
                await ClockPurePython._sleep_ns(wait_time)
                slew['asyncio.sleep'] = wait_time - start_timer
                slew['previous_loop_adjust'] = - (time.perf_counter_ns() - slew['asyncio.sleep'])

                # Advance to the next tick, the time offset should be adjusted at the next sleep
                self._current_tick = self._current_tick + interval_to_next_tick

                # Recording cycle duration
                slew['cycle'] = time.perf_counter_ns()
        finally:
            [ci.stop(self) for ci in self._current_context]

    async def run_till_in_timers(self, timestamp: float):
        if self._current_context is None:
            self.logger().error("run() and run_til() can only be used within the context of a `with...` statement.")
            raise EnvironmentError("run() and run_til() can only be used within the context of a `with...` statement.")

        self._construct_iterators_timetable()

        slew: Dict[str, Union[int, List[Dict[str, int]]]] = {'asyncio.sleep': 0,
                                                             'total_cycle': 0,
                                                             'iterators': [],
                                                             'iterators_cycle': 0,
                                                             'previous_loop_adjust': 0,
                                                             }
        try:
            # Starts the network, connector, exchange and timers
            self._start_clocking_in_context()
            # Monitors how everything clocks and terminates at the given timestamp
            while True:
                # Exit condition - Using finally to finish cleanly
                if self._current_tick > timestamp:
                    return

                self.logger().info(
                    f"Iterators all green:{all([i.is_alive() for i in self._ticks_cycle['iterators_timers']])}")
                wait_time = next(self._ticks_cycle['tick_intervals'])

                start_timer = time.perf_counter_ns()
                await ClockPurePython._async_sleep_ns(wait_time - wait_time)
                slew['asyncio.sleep'] = wait_time - start_timer
                slew['previous_loop_adjust'] = - (time.perf_counter_ns() - slew['asyncio.sleep'])

                # Advance to the next tick, the time offset should be adjusted at the next sleep
                self._current_tick = self._current_tick + wait_time

                # Recording cycle duration
                slew['cycle'] = time.perf_counter_ns()
        finally:
            # Stopping all the timers
            [t.cancel() for t in self._ticks_cycle['iterators_timers']]
            # Stopping all the iterators
            [ci.stop(self) for ci in self._current_context]

    async def run_till_gather(self, timestamp: float):
        if self._current_context is None:
            self.logger().error("run() and run_til() can only be used within the context of a `with...` statement.")
            raise EnvironmentError("run() and run_til() can only be used within the context of a `with...` statement.")

        self._construct_iterators_timetable()

        slew: Dict[str, Union[int, List[Dict[str, int]]]] = {'asyncio.sleep': 0,
                                                             'total_cycle': 0,
                                                             'iterators': [],
                                                             'iterators_cycle': 0,
                                                             'previous_loop_adjust': 0,
                                                             }
        try:
            # Starts the network, connector, exchange and timers
            self._start_clocking_in_context()
            # Monitors how everything clocks and terminates at the given timestamp
            while True:
                # Exit condition - Using finally to finish cleanly
                if self._current_tick > timestamp:
                    return

                self.logger().info(
                    f"Iterators all green:{all([i.is_alive() for i in self._ticks_cycle['iterators_timers']])}")
                wait_time = next(self._ticks_cycle['tick_intervals'])

                start_timer = time.perf_counter_ns()
                await ClockPurePython._async_sleep_ns(wait_time - wait_time)
                slew['asyncio.sleep'] = wait_time - start_timer
                slew['previous_loop_adjust'] = - (time.perf_counter_ns() - slew['asyncio.sleep'])

                # Advance to the next tick, the time offset should be adjusted at the next sleep
                self._current_tick = self._current_tick + wait_time

                # Recording cycle duration
                slew['cycle'] = time.perf_counter_ns()
        finally:
            # Stopping all the timers
            [t.cancel() for t in self._ticks_cycle['iterators_timers']]
            # Stopping all the iterators
            [ci.stop(self) for ci in self._current_context]

    def backtest_til(self, timestamp: float):
        if not self._started:
            for ci in self._child_iterators:
                child_iterator = ci
                child_iterator.start(self, self._start_time)
            self._started = True

        try:
            while not (self._current_tick >= timestamp):
                self._current_tick += self._tick_size
                for ci in self._child_iterators:
                    child_iterator = ci
                    try:
                        child_iterator.tick(self._current_tick)
                    except StopIteration:
                        raise
                    except Exception:
                        self.logger().error("Unexpected error running clock tick.", exc_info=True)
        except StopIteration:
            return
        finally:
            for ci in self._child_iterators:
                child_iterator = ci
                child_iterator.stop(self)

    def backtest(self):
        self.backtest_til(self._end_time)
