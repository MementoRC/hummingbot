from __future__ import annotations

import asyncio
import ctypes
import functools
import itertools
import logging
import os
import threading
import time
import typing
from collections import OrderedDict, defaultdict
from decimal import Decimal
from functools import reduce
from math import gcd
from typing import Dict, List, Set, Union

from hummingbot.core.clock_mode import ClockMode
from hummingbot.core.utils.async_utils import EventThreadSafe
from hummingbot.logger import HummingbotLogger

if typing.TYPE_CHECKING:
    from hummingbot.core.time_iterator_pp import TimeIteratorPurePython

s_logger = None


def set_tick_event(event: EventThreadSafe):
    event.set()


def clear_tick_event(event: EventThreadSafe):
    event.clear()


class LocalTimer(threading.Timer):
    result: List[int] = None

    def __init__(self, interval: float, function: callable, *args, **kwargs):
        if kwargs is None:
            kwargs = {}
        if args is None:
            args = []
        self._original_function = function
        self._interval = interval
        if LocalTimer.result is None:
            LocalTimer.result = list(args[0])

        if "iterations" not in kwargs:
            if len(args) > 1:
                kwargs["iterations"] = args[1]
            else:
                kwargs["iterations"] = 0

        if kwargs["iterations"] == 0:
            super(LocalTimer, self).__init__(
                self._interval, self._do_execution, args, kwargs)
        else:
            super(LocalTimer, self).__init__(
                self._interval, self._do_recursion, args, kwargs)

    def _do_recursion(self, *a, **kw):
        LocalTimer.result[kw["iterations"] - 1] = self._original_function()
        if kw["iterations"] != 1:
            kw["iterations"] = kw["iterations"] - 1
            LocalTimer(self._interval,
                       self._original_function,
                       LocalTimer.result,
                       iterations=kw['iterations']).start()
        else:
            LocalTimer.result[kw["iterations"] - 1] = self._original_function()

    def _do_execution(self, *a, **kw):
        LocalTimer.result[kw["iterations"]] = self._original_function()

    def join(self):
        super(LocalTimer, self).join()
        return LocalTimer.result


def print_delta_time() -> int:
    start = time.perf_counter_ns()
    return time.perf_counter_ns() - start


class MyThread(threading.Thread):
    def __init__(self, event):
        threading.Thread.__init__(self)
        self.stopped = event

    def run(self):
        while not self.stopped.wait(0.5):
            print("my thread")
            # call a function


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


class HighPrecisionWallTime:
    def __init__(self, ):
        self._wall_time_0 = time.time_ns()
        self._clock_0 = time.perf_counter_ns()

    def sample(self, ):
        dc = time.perf_counter_ns() - self._clock_0
        return self._wall_time_0 + dc


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
        self._list_primary_ticks = None
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

        self._current_tick_size = self._tick_size
        self._current_tick = self._start_time if clock_mode is ClockMode.BACKTEST else self._set_current_tick()

    @property
    def clock_mode(self) -> ClockMode:
        return self._clock_mode

    @property
    def start_time(self) -> float:
        return ns_s(self._start_time)

    @property
    def tick_size(self) -> float:
        return ns_s(self._tick_size)

    @property
    def child_iterators(self) -> List["TimeIteratorPurePython"]:
        return self._child_iterators

    @property
    def current_timestamp_ns(self) -> int:
        return self._current_tick

    @property
    def current_timestamp(self) -> float:
        return ns_s(self._current_tick)

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
        self._current_context = None

    def add_iterator(self, iterator: "TimeIteratorPurePython"):
        if self._current_context is not None:
            self._current_context.append(iterator)
        if self._started:
            iterator.start(self, self._current_tick)
        self._child_iterators.append(iterator)

    def remove_iterator(self, iterator: "TimeIteratorPurePython"):
        if self._current_context is not None and iterator in self._current_context:
            iterator.stop(self)
            self._current_context.remove(iterator)
        self._child_iterators.remove(iterator)

    def _start_clocking_in_context(self) -> int:
        if self._current_context is None:
            self.logger().error("Context is not initialized")
            raise EnvironmentError("context is not initialized")
        # Start the clock to pass to the iterators
        self._set_current_tick()
        # Start iterators
        if not self._started:
            [ci.start(self) for ci in self._current_context]
            self._started = True

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
        base_intervals = sorted(list(set([i.tick_size for i in self._current_context])))
        # Calculate the lowest common multiple of the primary tick_sizes, i.e base_intervals
        last_tick_time = reduce(lambda a, b: a * b // gcd(a, b), base_intervals)
        # All iterators start with the Clock (start time is not know at initialization)
        dict_iterators[0] |= set(self._current_context)

        # The dict automatically removes duplicated time and simply add the needed iterators
        #   At the end the dict is replaced by a OrderedDict
        for base_interval in base_intervals:
            # Initialize the tick_time to t=0 + base_interval
            tick_time = base_interval
            # Collect the list of iterators with tick_size == tick_interval
            interval_iterators = {ci for ci in self._current_context if ci.tick_size == base_interval}
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
        # List of the succession of intervals between the Clock ticks
        self._ticks_cycle['tick_intervals'] = [j - i for i, j in
                                               zip(list(it_cycle.keys())[:-1], list(it_cycle.keys())[1:])]
        self._ticks_cycle['intervals_cycle_iterator'] = itertools.cycle(self._ticks_cycle['tick_intervals'])

    def _create_timers(self):
        """
        Creates the Threading Timers that will periodically execute each iterators list
        """

    @staticmethod
    def _tick_formula(time_in_ns: int, tick_size_in_ns: int, is_next: bool = False) -> int:
        d = (time_in_ns // tick_size_in_ns + int(is_next)) * tick_size_in_ns
        return d

    def _set_current_tick(self) -> int:
        if self._ticks_cycle and self._ticks_cycle['tick_intervals']:
            self._current_tick_size = next(self._ticks_cycle['tick_intervals'])
        self._current_tick = ClockPurePython._tick_formula(time.time_ns(), self._current_tick_size, False)
        return self._current_tick

    async def run(self):
        await self.run_til(float("nan"))

    @staticmethod
    def _print_time():
        print(time.perf_counter())

    @staticmethod
    def _spleep_ns(delay: int, offset: int = 1500, monitor=None, timer_ns: callable = time.monotonic_ns) -> Dict[str, int]:
        """
        Busy Wait timer. The minimum delta measurable on my machine it 1
        """
        class timespec(ctypes.Structure):
            _fields_ = \
                [
                    ('tv_sec', ctypes.c_long),
                    ('tv_nsec', ctypes.c_long)
                ]

        librt = ctypes.CDLL('librt.so.1', use_errno=True)

        clock_gettime = librt.clock_gettime
        clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(timespec)]

        c_clock_nanosleep = librt.clock_nanosleep
        c_clock_nanosleep.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(timespec), ctypes.POINTER(timespec)]

        def monotonic_time():
            "return a timestamp in seconds (sec)"
            t = timespec()
            # (Note that clock_gettime() returns 0 for success, or -1 for failure, in
            # which case errno is set appropriately)
            # -see here: http://linux.die.net/man/3/clock_gettime
            if clock_gettime(4, ctypes.pointer(t)) != 0:
                # if clock_gettime() returns an error
                errno_ = ctypes.get_errno()
                raise OSError(errno_, os.strerror(errno_))

            return t.tv_sec + t.tv_nsec

        def clock_nanosleep(sleep: int):
            "return a timestamp in seconds (sec)"
            request = timespec()
            # clock_gettime(4, ctypes.pointer(request))
            request.tv_nsec = sleep
            # if request.tv_nsec >= 1000000000:
            #    request.tv_nsec = request.tv_nsec - 1000000000
            #    request.tv_sec = request.tv_sec + 1

            (c_clock_nanosleep(time.CLOCK_MONOTONIC, 0, ctypes.pointer(request), None))

        if monitor is None:
            monitor = []
        if delay < 700:
            start: int = timer_ns()
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
        time.time_ns()
        start: int = timer_ns()

        clock_nanosleep(delay - offset // 2)
        # while end - timer_ns() >= 0:
        #    for i in range(0, len(monitor)):
        #        # time.time_ns()
        #        t0 = timer_ns()
        #        #clock_nanosleep(1000)
        #        monitor[i] = timer_ns() - t0

        current_time = timer_ns()
        wait = current_time - start
        residual = wait - delay
        return {"current_time": current_time,
                "busy_wait": wait,
                "busy_residual": residual,
                "effective_wait": wait,
                "residual": residual,
                "count": count,
                "changes": monitor,
                }

    @staticmethod
    def _spin_5us_ns(delay: int, accuracy: int = 10000, timer_ns: callable = time.perf_counter_ns) -> Dict[str, int]:
        """
        Busy Wait timer. The minimum delta measurable on my machine it 1
        """
        if delay < accuracy // 2:
            start: int = timer_ns()
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
        time.time_ns()
        start: int = timer_ns()
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
                "count": count,
                "changes": [],
                }

    @staticmethod
    def _spin_500us_ns(delay: int, busy_offset: int = 700) -> Dict[str, int]:
        start: int = time.perf_counter_ns()
        timing_data: Dict[str, int] = dict(sleep0_wait=0,
                                           sleep1_wait=0,
                                           )
        if delay < 1000:
            return {"current_time": start,
                    "busy_wait": 0,
                    "busy_residual": 0,
                    "effective_wait": 0,
                    "residual": 0,
                    "sleep0_wait": 0,
                    "sleep1_wait": 0,
                    }
        elif delay < 85000:
            wait: Dict[str, int] = ClockPurePython._spin_5us_ns(delay)
            wait.update(timing_data)
            return wait

        # The sleep(0) is equivalent "busy_wait"
        elif delay < 800000:
            min_sleep_time: float = 0
            sleep_offset: int = 6000
        # The minimum sleep time on my machine was 80-160us
        else:
            min_sleep_time: float = 1e-9
            sleep_offset: int = 400000

        end: int = start + delay - sleep_offset
        while True:
            time.sleep(min_sleep_time)
            monitor = time.perf_counter_ns()
            if monitor >= end:
                if min_sleep_time == 0:
                    timing_data['sleep0_wait'] = monitor - start
                else:
                    timing_data['sleep1_wait'] = monitor - start
                end: int = start + delay
                while True:
                    wait = ClockPurePython._spin_5us_ns(sleep_offset)
                    if wait['current_time'] >= end:
                        wait['effective_wait'] = wait['current_time'] - start
                        wait['residual'] = wait['effective_wait'] - delay
                        wait.update(timing_data)
                        return wait

    @staticmethod
    async def _sleep_ns(delay: int, async_threshold: int = 30000000, async_accuracy: int = 10000000) -> Dict[str, int]:
        start_effective = time.perf_counter_ns()
        sync_wait, async_wait = 0, 0
        if delay < async_threshold:
            ClockPurePython.logger().warning(f"Asyncio.sleep({ns_s(delay)}s/{delay}ns) requested too small. {50}ms "
                                             f"accuracy. Using blocking time.sleep().")
            wait = ClockPurePython._spin_500us_ns(delay)
        else:
            start = time.perf_counter_ns()
            await asyncio.sleep(ns_s(delay - async_accuracy))
            async_wait = time.perf_counter_ns() - start
            wait = ClockPurePython._spin_500us_ns(async_wait - delay)
        wait['sync_wait'] = sync_wait
        wait['async_wait'] = async_wait
        wait['effective_wait'] = time.perf_counter_ns() - start_effective
        wait['residual'] = wait['effective_wait'] - delay
        return wait

    def _profile(self, func):
        def call_f(*args, **kwargs):
            self.pr.enable()
            returned_value = func(*args, **kwargs)
            self.pr.disable()
            return returned_value

        return call_f

    async def run_til(self, timestamp: float):
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
