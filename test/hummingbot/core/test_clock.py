from __future__ import annotations

import asyncio
import cProfile
import inspect
import threading
import time
import unittest
import warnings
from functools import reduce
from math import sqrt
from pathlib import Path
from pprint import pprint
from pstats import Stats
from typing import Dict
from unittest.mock import patch

import matplotlib
import numpy
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt
from scipy.stats import describe
from scipy.stats._continuous_distns import _distn_names

from hummingbot import root_path
from hummingbot.core.clock_pp import ClockMode, ClockPurePython, ns_s, s_ns
from hummingbot.core.time_iterator import TimeIterator
from hummingbot.core.time_iterator_pp import TimeIteratorPurePython

log_records = {'realtime': [], 'backtest': []}


class HandleRealtime:
    level = 0

    def handle(self, record):
        global log_records
        log_records['realtime'].append(record)


def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = numpy.histogram(data, bins=bins, density=True)
    x = (x + numpy.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if d not in ['levy_stable', 'studentized_range']]):

        print("{:>3} / {:<3}: {}".format(ii + 1, len(_distn_names), distribution))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)
                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = numpy.sum(numpy.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))

        except Exception:
            pass

    return sorted(best_distributions, key=lambda x: x[2])


def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


class HandleBacktest:
    level = 0

    def handle(self, record):
        global log_records
        log_records['backtest'].append(record)


class ClockUnitTest(unittest.TestCase):
    # logging.Level required to receive logs from the data source logger
    level = 0

    backtest_start_timestamp: float = pd.Timestamp("2021-01-01", tz="UTC").timestamp()
    backtest_end_timestamp: float = pd.Timestamp("2021-01-01 01:00:00", tz="UTC").timestamp()
    tick_size: float = 1
    data_dir: Path = root_path() / "test" / "profiling-data" / __file__.split('test/')[1].split('.')[0]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ev_loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        cls.data_dir.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        super().setUp()
        self.realtime_start_timestamp = time.time()
        self.realtime_end_timestamp = self.realtime_start_timestamp + 2.0
        self.clock_realtime = ClockPurePython(ClockMode.REALTIME, self.tick_size, self.realtime_start_timestamp,
                                              self.realtime_end_timestamp)
        self.clock_backtest = ClockPurePython(ClockMode.BACKTEST, self.tick_size, self.backtest_start_timestamp,
                                              self.backtest_end_timestamp)

        # Logger
        self.clock_realtime.logger().setLevel(1)
        self.clock_realtime.logger().addHandler(HandleRealtime())
        self.clock_backtest.logger().setLevel(1)
        self.clock_backtest.logger().addHandler(HandleBacktest())

        # Profiler
        self.pr = cProfile.Profile()
        self.current_test = None

    def tearDown(self):
        """finish any test"""
        try:
            p = Stats(self.pr)
        finally:
            pass
        p.strip_dirs()
        p.sort_stats('cumtime')
        p.print_stats()
        p.dump_stats(f"{self.data_dir}/{self.current_test}.pstats")
        print("\n--->>>")

    @staticmethod
    def check_is_logged(clock: str, log_level: str, message: str) -> bool:
        return any(
            record.levelname == log_level and record.getMessage() == message
            for record in log_records[clock]
        )

    def call_with_profile(self, func):
        def call_f(*args, **kwargs):
            self.current_test = inspect.currentframe().f_back.f_code.co_name
            print(f"Profiling {func.__name__}")
            self.pr.enable()
            returned_value = func(*args, **kwargs)
            self.pr.disable()
            return returned_value

        return call_f

    def test_clock_mode(self):
        self.assertEqual(ClockMode.REALTIME, self.clock_realtime.clock_mode)
        self.assertEqual(ClockMode.BACKTEST, self.clock_backtest.clock_mode)

    def test_start_time(self):
        self.assertEqual(self.realtime_start_timestamp, self.clock_realtime.start_time)
        self.assertEqual(self.backtest_start_timestamp, self.clock_backtest.start_time)

    def test_tick_size(self):
        self.assertEqual(self.realtime_start_timestamp, self.clock_realtime.start_time)
        self.assertEqual(self.backtest_start_timestamp, self.clock_backtest.start_time)

    def test_child_iterators(self):
        # Tests child_iterators property after initialization. See also test_add_iterator
        self.assertEqual(0, len(self.clock_realtime.child_iterators))
        self.assertEqual(0, len(self.clock_backtest.child_iterators))

    def test_current_timestamp(self):
        time_ns = 1234567890000000000
        tick_s = 1e-9 * (time_ns // self.tick_size) * self.tick_size
        self.clock_realtime._current_tick = 0
        self.clock_backtest._current_tick = 0
        with patch.object(time, 'time_ns') as mocked_time:
            mocked_time.return_value = time_ns
            self.clock_realtime._set_current_tick()
            self.clock_backtest._set_current_tick()
            self.assertAlmostEqual(tick_s, self.clock_realtime.current_timestamp)
            self.assertAlmostEqual(tick_s, self.clock_backtest.current_timestamp)

    def test_current_timestamp_ns(self):
        time_ns = 1234567890000000000
        self.clock_realtime._current_tick = 0
        self.clock_backtest._current_tick = 0
        with patch.object(time, 'time_ns') as mocked_time:
            mocked_time.return_value = time_ns
            self.clock_realtime._set_current_tick()
            self.clock_backtest._set_current_tick()
            self.assertAlmostEqual(time_ns, self.clock_realtime.current_timestamp_ns)
            self.assertAlmostEqual(time_ns, self.clock_backtest.current_timestamp_ns)

    def test__set_current_tick(self):
        time_ns = 1234567890000000000
        self.clock_realtime._current_tick = 0
        self.clock_backtest._current_tick = 0
        with patch.object(time, 'time_ns') as mocked_time:
            mocked_time.return_value = time_ns
            real_tick = self.call_with_profile(self.clock_realtime._set_current_tick)()
            back_tick = self.clock_backtest._set_current_tick()
            self.assertEqual(time_ns, self.clock_realtime._current_tick)
            self.assertEqual(time_ns, self.clock_backtest._current_tick)
            self.assertEqual(time_ns, real_tick)
            self.assertEqual(time_ns, back_tick)

    def test__tick_formula(self):
        time_in_ns = 1234567890000000000
        tick_period_in_ns = 1000000000
        # Tick for a given time, period
        self.assertEqual(time_in_ns, self.clock_realtime._tick_formula(time_in_ns, tick_period_in_ns, False))
        # 1ns within the tick period
        self.assertEqual(time_in_ns, self.clock_realtime._tick_formula(time_in_ns + 1, tick_period_in_ns, False))
        # 1ns before the end of the tick period
        self.assertEqual(time_in_ns,
                         self.clock_realtime._tick_formula(time_in_ns + tick_period_in_ns - 1, tick_period_in_ns,
                                                           False))
        # 1ns after the end of the tick period -> next tick = time + tick period
        self.assertEqual(time_in_ns + tick_period_in_ns,
                         self.clock_realtime._tick_formula(time_in_ns + tick_period_in_ns + 1, tick_period_in_ns,
                                                           False))
        # Next tick for the given time and tick period -> time + tick period
        self.assertEqual(time_in_ns + tick_period_in_ns,
                         self.clock_realtime._tick_formula(time_in_ns, tick_period_in_ns, True))
        # Next tick at 1ns before the end of the tick period
        self.assertEqual(time_in_ns + tick_period_in_ns,
                         self.clock_realtime._tick_formula(time_in_ns + tick_period_in_ns - 1, tick_period_in_ns, True))

    def test_add_iterator(self):
        self.assertEqual(0, len(self.clock_realtime.child_iterators))
        self.assertEqual(0, len(self.clock_backtest.child_iterators))

        time_iterator: TimeIterator = TimeIterator()
        self.clock_realtime.add_iterator(time_iterator)
        self.clock_backtest.add_iterator(time_iterator)

        self.assertEqual(1, len(self.clock_realtime.child_iterators))
        self.assertEqual(time_iterator, self.clock_realtime.child_iterators[0])
        self.assertEqual(1, len(self.clock_backtest.child_iterators))
        self.assertEqual(time_iterator, self.clock_backtest.child_iterators[0])

    def test_remove_iterator(self):
        self.assertEqual(0, len(self.clock_realtime.child_iterators))
        self.assertEqual(0, len(self.clock_backtest.child_iterators))

        time_iterator: TimeIterator = TimeIterator()
        self.clock_realtime.add_iterator(time_iterator)
        self.clock_backtest.add_iterator(time_iterator)

        self.assertEqual(1, len(self.clock_realtime.child_iterators))
        self.assertEqual(time_iterator, self.clock_realtime.child_iterators[0])
        self.assertEqual(1, len(self.clock_backtest.child_iterators))
        self.assertEqual(time_iterator, self.clock_backtest.child_iterators[0])

        self.clock_realtime.remove_iterator(time_iterator)
        self.clock_backtest.remove_iterator(time_iterator)

        self.assertEqual(0, len(self.clock_realtime.child_iterators))
        self.assertEqual(0, len(self.clock_backtest.child_iterators))

    def test__start_clock_in_context_raises_exception(self):
        self.assertEqual(None, self.clock_realtime._current_context)
        with self.assertRaises(EnvironmentError):
            self.clock_realtime._start_clocking_in_context()

        self.assertEqual(None, self.clock_backtest._current_context)
        with self.assertRaises(EnvironmentError):
            self.clock_backtest._start_clocking_in_context()

    def test__start_clock_in_context(self):
        time_in_ns = 1234567890000000000
        self.assertNotEqual(time_in_ns, self.clock_realtime._current_tick)
        self.assertNotEqual(time_in_ns, self.clock_backtest._current_tick)
        with patch.object(time, 'time_ns') as mocked_time:
            with patch.object(TimeIteratorPurePython, 'start') as mocked_start:
                mocked_time.return_value = time_in_ns
                self.assertEqual(False, self.clock_realtime._started)
                self.clock_realtime._current_context = [TimeIteratorPurePython()]

                self.call_with_profile(self.clock_realtime._start_clocking_in_context)()

                self.assertEqual(True, self.clock_realtime._started)
                self.assertEqual(time_in_ns, self.clock_realtime._current_tick)
                mocked_start.assert_called_with(self.clock_realtime)

        with patch.object(time, 'time_ns') as mocked_time:
            with patch.object(TimeIteratorPurePython, 'start') as mocked_start:
                mocked_time.return_value = time_in_ns
                self.assertEqual(False, self.clock_backtest._started)
                self.clock_backtest._current_context = [TimeIteratorPurePython()]

                self.clock_backtest._start_clocking_in_context()

                self.assertEqual(True, self.clock_backtest._started)
                self.assertEqual(time_in_ns, self.clock_backtest._current_tick)
                mocked_start.assert_called_with(self.clock_backtest)

    def test__construct_iterators_timetable_raise_no_context(self):
        self.assertEqual(None, self.clock_realtime._current_context)
        with self.assertRaises(EnvironmentError) as e:
            self.clock_realtime._construct_iterators_timetable()

        log_target = "Context is not initialized"
        self.assertEqual(log_target, str(e.exception))
        self.assertTrue(self.check_is_logged(clock='realtime', log_level="ERROR", message=log_target))

        self.assertEqual(None, self.clock_backtest._current_context)
        with self.assertRaises(EnvironmentError) as e:
            self.clock_backtest._construct_iterators_timetable()
        self.assertEqual(log_target, str(e.exception))
        self.assertTrue(self.check_is_logged(clock='backtest', log_level="ERROR", message=log_target))

    def test__construct_iterators_timetable_raise_context_empty(self):
        self.assertEqual(None, self.clock_realtime._current_context)
        self.clock_realtime._current_context = []
        with self.assertRaises(EnvironmentError) as e:
            self.clock_realtime._construct_iterators_timetable()

        log_target = "Context is empty"
        self.assertEqual(log_target, str(e.exception))
        self.assertTrue(self.check_is_logged(clock='realtime', log_level="ERROR", message=log_target))

        self.assertEqual(None, self.clock_backtest._current_context)
        self.clock_backtest._current_context = []
        with self.assertRaises(EnvironmentError) as e:
            self.clock_backtest._construct_iterators_timetable()
        self.assertEqual(log_target, str(e.exception))
        self.assertTrue(self.check_is_logged(clock='backtest', log_level="ERROR", message=log_target))

    def test__construct_iterators_timetable_same_base(self):
        list_ticks = [100, 200]
        list_iterators = []
        self.clock_realtime._current_context = []
        self.clock_backtest._current_context = []
        for ts in list_ticks:
            time_iterator: TimeIteratorPurePython = TimeIteratorPurePython()
            time_iterator.tick_size = ts
            list_iterators.append(time_iterator)
            self.clock_realtime._current_context.append(time_iterator)
            self.clock_backtest._current_context.append(time_iterator)

        # Expected result, start at 0, ticks every 100s, 200s -> lower common multiple is 200s
        expected_list_intervals = [100]

        self.clock_realtime._construct_iterators_timetable()

        self.assertEqual(sorted(['tick_intervals', 'iterators_cycle', 'intervals_cycle_iterator']),
                         sorted(self.clock_realtime._ticks_cycle.keys()))
        self.assertEqual(len(expected_list_intervals), len(self.clock_realtime._ticks_cycle['tick_intervals']))
        self.assertEqual(expected_list_intervals, self.clock_realtime._ticks_cycle['tick_intervals'])
        self.assertTrue(
            all([k % vi.tick_size == 0 for k, v in self.clock_realtime._ticks_cycle['iterators_cycle'].items() for vi in
                 v]))
        self.assertTrue(
            all([i in {v for av in self.clock_realtime._ticks_cycle['iterators_cycle'].values() for v in av} for i in
                 list_iterators]))
        self.assertTrue(
            all([v in list_iterators for av in self.clock_realtime._ticks_cycle['iterators_cycle'].values() for v in
                 av]))

    def test__construct_iterators_timetable_prime(self):
        list_ticks_size = [300, 200]
        list_iterators = []
        self.clock_realtime._current_context = []
        self.clock_backtest._current_context = []
        for ts in list_ticks_size:
            time_iterator: TimeIteratorPurePython = TimeIteratorPurePython()
            time_iterator.tick_size = ts
            list_iterators.append(time_iterator)
            self.clock_realtime._current_context.append(time_iterator)
            self.clock_backtest._current_context.append(time_iterator)

        # Expected result, start at 0, ticks every 100s, 200s -> lower common multiple is 200s
        # expected_list_ticks = [0, 200, 300, 400, 600]
        expected_list_intervals = [200, 100, 100, 200]

        self.clock_realtime._construct_iterators_timetable()

        self.assertEqual(sorted(['tick_intervals',
                                 'iterators_cycle',
                                 'timer_for_iterator',
                                 'intervals_cycle_iterator',
                                 'iterators_timers']),
                         sorted(self.clock_realtime._ticks_cycle.keys()))
        self.assertEqual(len(expected_list_intervals), len(self.clock_realtime._ticks_cycle['tick_intervals']))
        self.assertEqual(expected_list_intervals, self.clock_realtime._ticks_cycle['tick_intervals'])
        self.assertTrue(
            all([k % vi.tick_size == 0 for k, v in self.clock_realtime._ticks_cycle['iterators_cycle'].items() for vi in
                 v]))
        self.assertTrue(
            all([i in [v for av in list(self.clock_realtime._ticks_cycle['iterators_cycle'].values()) for v in av] for i
                 in
                 list_iterators]))
        self.assertTrue(
            all([v in list_iterators for av in self.clock_realtime._ticks_cycle['iterators_cycle'].values() for v in
                 av]))
        self.assertEqual(len(list_ticks_size), len(self.clock_realtime._ticks_cycle['iterators_timers']))
        self.assertEqual(sorted([ns_s(i) for i in list_ticks_size]),
                         sorted([i.interval for i in self.clock_realtime._ticks_cycle['iterators_timers']]))
        pprint(vars(self.clock_realtime._ticks_cycle['iterators_timers'][-1]))

    def test_bar(self):
        foo = threading.Event()
        with patch('spam.foo', side_effect=lambda *args, **kwargs: foo.set()) as mock:
            # Make the callback `foo` to be called immediately
            with patch.object(threading._Event, 'wait', time.sleep(0.000001)):
                pass
            foo.wait()  # Wait until `spam.foo` is called. (instead of time.sleep)
            mock.assert_called_once_with('potato', y='spam', x=69)

    def test__busy_wait_ns(self):
        import matplotlib.pyplot as plt
        from scipy.stats import describe

        class LocalThread(threading.Thread):
            def __init__(self, target):
                super().__init__(None, target)
                self.result = wait
                self.target = target

            def run(self):
                x = sqrt(3)
                x = x + 1
                for i in range(nb_samples):
                    wait[i] = self.target(200000)

        nb_samples = 20000
        wait = [-1] * nb_samples
        t = LocalThread(target=self.clock_realtime._busy_wait_ns)
        t.start()
        t.join()
        # x = [self.clock_realtime._spin_5us_ns(5000, 0) for i in range(nb_samples)]
        data = pd.Series([i['effective_wait'] for i in t.result])
        plt.figure(figsize=(12, 8))
        ax = data.plot(kind='hist', bins=100, density=True, alpha=0.5,
                       color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

        # Save plot limits
        dataYLim = ax.get_ylim()

        # Find best fit distribution
        best_distibutions = best_fit_distribution(data, 200, ax)
        best_dist = best_distibutions[0]

        # Update plots
        ax.set_ylim(dataYLim)
        ax.set_title(u'Distribution of delays\n All Fitted Distributions')
        ax.set_xlabel(u'Delay (ns)')
        ax.set_yscale('log')
        ax.set_ylabel('Frequency')

        # Make PDF with best params
        pdf = make_pdf(best_dist[0], best_dist[1])

        # Display
        plt.figure(figsize=(12, 8))
        ax = pdf.plot(lw=2, label='PDF', legend=True)
        data.plot(kind='hist', bins=100, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

        param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_dist[1])])
        dist_str = '{}({})'.format(best_dist[0].name, param_str)

        ax.set_title(u'Distribution of delays. with best fit distribution \n' + dist_str)
        ax.set_xlabel(u'Temp. (°C)')
        ax.set_yscale('log')
        ax.set_ylabel('Frequency')
        plt.show()

        print(describe([(i['effective_wait'] - 200000) / 200000 * 100 for i in t.result]))
        # print(f"{min([i['count'] for i in x])} {max([i['count'] for i in x])}")
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0, 0].hist([i['effective_wait'] for i in t.result], bins=100, linewidth=1, histtype='step', density=False,
                      cumulative=False)
        ax[0, 0].set_yscale('log')
        ax[0, 1].hist([i['effective_wait'] for i in t.result], bins=100, linewidth=1, histtype='step', density=True,
                      cumulative=True)
        ax[1, 0].hist([i['busy_count'] for i in t.result], bins=100, linewidth=1, histtype='step', density=False,
                      cumulative=False)
        ax[1, 0].set_yscale('log')
        ax[1, 1].hist([i['busy_count'] for i in t.result], bins=100, linewidth=1, histtype='step', density=True,
                      cumulative=True)
        plt.show()

        mean = (reduce(lambda a, b: a + b, [i['effective_wait'] for i in t.result], 0) / nb_samples)
        error = (reduce(lambda a, b: a + b, [i['residual'] for i in t.result], 0) / nb_samples)
        print(f"5us Clock: {mean} {error:+}ns -> {100 * error / mean:.2}%")

    def test__time_sleep_ns(self):
        import matplotlib.pyplot as plt
        from scipy.stats import describe

        class LocalThread(threading.Thread):
            def __init__(self, target):
                super().__init__(None, target)
                self.result = wait
                self.target = target

            def run(self):
                x = sqrt(3)
                x = x + 1
                for i in range(nb_samples):
                    wait[i] = self.target(delay, busy_wait=True)

        fig, ax = plt.subplots(nrows=2, ncols=2)
        nb_samples = 1000
        delay = 10000000
        wait = [-1] * nb_samples
        t = LocalThread(target=self.clock_realtime._time_sleep_ns)
        t.start()
        t.join()

        # x = [self.clock_realtime._spin_5us_ns(5000, 0) for i in range(nb_samples)]
        print(describe([(i['effective_wait'] - delay) / delay * 100 for i in t.result]))
        print(describe([(i['sleep_count']) for i in t.result]))
        print(describe([(i['sleep0_wait']) / delay * 100 for i in t.result]))
        print(describe([(i['sleep1_wait']) / delay * 100 for i in t.result]))
        print(describe([(i['busy_wait']) / delay * 100 for i in t.result]))
        print(describe([(i['busy_count']) for i in t.result]))

        # print(f"{min([i['count'] for i in x])} {max([i['count'] for i in x])}")
        ax[0, 0].hist([i['effective_wait'] for i in t.result], bins=100, linewidth=1, histtype='step', density=False,
                      cumulative=False)
        ax[0, 0].set_yscale('log')
        ax[0, 1].hist([i['effective_wait'] for i in t.result], bins=100, linewidth=1, histtype='step', density=True,
                      cumulative=True)
        ax[1, 0].hist([i['busy_count'] for i in t.result], bins=100, linewidth=1, histtype='step', density=False,
                      cumulative=False)
        ax[1, 0].hist([i['sleep_count'] for i in t.result], bins=100, linewidth=1, histtype='step', density=False,
                      cumulative=False)
        ax[1, 0].set_yscale('log')
        ax[1, 1].hist([i['busy_count'] for i in t.result], bins=100, linewidth=1, histtype='step', density=True,
                      cumulative=True)
        plt.show()

        mean = (reduce(lambda a, b: a + b, [i['effective_wait'] for i in t.result], 0) / nb_samples)
        error = (reduce(lambda a, b: a + b, [i['residual'] for i in t.result], 0) / nb_samples)
        print(f"5us Clock: {mean} {error:+}ns -> {100 * error / mean:.2}%")

    def test__sleep_ns_too_small(self):
        for delay in [1]:
            with patch.object(asyncio, 'sleep') as mocked_sleep:
                wait_data = self.ev_loop.run_until_complete(
                    self.clock_realtime._async_sleep_ns(delay, block_sleep=False))
                print(f"Waited: {wait_data['effective_wait']} ({100 * wait_data['residual'] / delay:.2}%)\n"
                      f"   Async wait: {wait_data['async_wait']} ({100 * wait_data['async_wait'] / delay:.2}%)\n"
                      f"   Sync wait: {wait_data['sync_wait']} ({100 * wait_data['sync_wait'] / delay:.2}%)"
                      )
                mocked_sleep.assert_not_called()
                self.assertNotEqual(delay * 1e-9, wait_data['effective_wait'] * 1e-9)
                log_target = f"Asyncio.sleep({ns_s(delay)}s/{delay}ns) requested too small. {50}ms accuracy. Using blocking time.sleep()."
                self.assertTrue(self.check_is_logged(clock='realtime', log_level="WARNING", message=log_target))
        for delay in [s_ns('0.001') - 1, s_ns('0.001')]:
            with patch.object(asyncio, 'sleep') as mocked_sleep:
                wait_data = self.ev_loop.run_until_complete(self.clock_realtime._async_sleep_ns(delay))
                print(f"Waited: {wait_data['effective_wait']} ({100 * wait_data['residual'] / delay:.2}%)\n"
                      f"   Async wait: {wait_data['async_wait']} ({100 * wait_data['async_wait'] / delay:.2}%)\n"
                      f"   Sync wait: {wait_data['sync_wait']} ({100 * wait_data['sync_wait'] / delay:.2}%)"
                      )
                mocked_sleep.assert_not_called()
                self.assertAlmostEqual(delay * 1e-9, wait_data['effective_wait'] * 1e-9, places=2)
                log_target = f"Asyncio.sleep({ns_s(delay)}s/{delay}ns) requested too small. {50}ms accuracy. Using blocking time.sleep()."
                self.assertTrue(self.check_is_logged(clock='realtime', log_level="WARNING", message=log_target))

    def test__sleep_ns_w_greedy(self):
        async def greedy_async():
            while True:
                k = 0
                for i in range(1 * 10 ** 6):
                    k += i
                await asyncio.sleep(0.1)  # yield to event loop

        async def tasks(sleep_time: int) -> Dict[str, int]:
            asyncio.create_task(greedy_async())  # start greedy coroutine
            while True:
                asyncio.create_task(asyncio.sleep(1))

                wait_data = await self.clock_realtime._async_sleep_ns(sleep_time,
                                                                      async_threshold=50000000,
                                                                      async_accuracy=1000000,
                                                                      block_sleep=False,
                                                                      throttle=True)

                return wait_data

        for delay in [s_ns('0.5')]:
            nb_samples = 5
            wait_data = [-1] * nb_samples
            for i in range(nb_samples):
                wait_data[i] = self.ev_loop.run_until_complete(tasks(delay))
            print(describe([(i['residual']) for i in wait_data]))
            fig, ax = plt.subplots(nrows=2, ncols=2)
            ax[0, 0].hist([i['effective_wait'] for i in wait_data], bins=100, linewidth=1, histtype='step',
                          density=False,
                          cumulative=False)
            # ax[0, 0].set_yscale('log')
            ax[0, 1].hist([i['effective_wait'] for i in wait_data], bins=100, linewidth=1, histtype='step',
                          density=True,
                          cumulative=True)
            plt.show()
            wait = self.ev_loop.run_until_complete(tasks(delay))
            print(f"Waited:\t\t\t{wait['effective_wait']:10} ({100 * wait['residual'] / delay:9.2f}%)\n"
                  f"    Busy wait:\t{wait['busy_wait']:10} ({100 * wait['busy_wait'] / delay:9.2f}%)\n"
                  f"  Sleep0 wait:\t{wait['sleep0_wait']:10} ({100 * wait['sleep0_wait'] / delay:9.2f}%)\n"
                  f"  Sleep1 wait:\t{wait['sleep1_wait']:10} ({100 * wait['sleep1_wait'] / delay:9.2f}%)\n"
                  f"   Async wait:\t{wait['async_wait']:10} ({100 * wait['async_wait'] / delay:9.2f}%)\n"
                  f"    Sync wait:\t{wait['sync_wait']:10} ({100 * wait['sync_wait'] / delay:9.2f}%)\n"
                  )
            # self.assertAlmostEqual(delay * 1e-9, wait * 1e-9, places=2)

#   def test_run(self):
#       # Note: Technically you do not execute `run()` when in BACKTEST mode

#       # Tests EnvironmentError raised when not running within a context
#       with self.assertRaises(EnvironmentError):
#           self.ev_loop.run_until_complete(self.clock_realtime.run())

#       # Note: run() will essentially run indefinitely hence the enforced timeout.
#       with self.assertRaises(asyncio.TimeoutError), self.clock_realtime:
#           self.ev_loop.run_until_complete(asyncio.wait_for(self.clock_realtime.run(), 1))

#       self.assertLess(self.realtime_start_timestamp, self.clock_realtime.current_timestamp)

#   def test_run_til(self):
#       # Note: Technically you do not execute `run_til()` when in BACKTEST mode

#       # Tests EnvironmentError raised when not runnning within a context
#       with self.assertRaises(EnvironmentError):
#           self.ev_loop.run_until_complete(self.clock_realtime.run_til(self.realtime_end_timestamp))

#       with self.clock_realtime:
#           self.ev_loop.run_until_complete(self.clock_realtime.run_til(self.realtime_end_timestamp))

#       self.assertGreaterEqual(self.clock_realtime.current_timestamp, self.realtime_end_timestamp)

#   def test_run_til_smaller_tick_size(self):
#       time_iterator: TimeIteratorPurePython = TimeIteratorPurePython()
#       self.clock_realtime.add_iterator(time_iterator)
#       tick_size = 0.1
#       time_iterator.tick_size = tick_size

#       with self.clock_realtime:
#           with patch.object(TimeIteratorPurePython, 'tick') as mocked_tick:
#               with patch.object(TimeIteratorPurePython, 'next_tick') as mocked_next_tick:
#                   mocked_next_tick.return_value = self.realtime_start_timestamp + tick_size
#                   self.ev_loop.run_until_complete(self.clock_realtime.run_til(self.realtime_end_timestamp))

#       mocked_next_tick.assert_called()
#       print(mocked_tick.call_args_list)
#       mocked_tick.assert_called_with(self.realtime_start_timestamp + tick_size)

#       self.assertEqual(tick_size, self.clock_realtime.current_timestamp - self.realtime_start_timestamp)

#   def test_backtest(self):
#       # Note: Technically you do not execute `backtest()` when in REALTIME mode

#       self.clock_backtest.backtest()
#       self.assertGreaterEqual(self.clock_backtest.current_timestamp, self.backtest_end_timestamp)

#   def test_backtest_til(self):
#       # Note: Technically you do not execute `backtest_til()` when in REALTIME mode

#       self.clock_backtest.backtest_til(self.backtest_start_timestamp + self.tick_size)
#       #self.assertGreater(self.clock_backtest.current_timestamp, self.clock_backtest.start_time)
#       #self.assertLess(self.clock_backtest.current_timestamp, self.backtest_end_timestamp)
