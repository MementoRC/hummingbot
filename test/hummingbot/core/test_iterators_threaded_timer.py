from __future__ import annotations

import cProfile
import inspect
import threading
import time
import unittest
from pathlib import Path
from pstats import Stats
from unittest.mock import call, patch

from hummingbot import root_path
from hummingbot.core.clock_pp import ClockPurePython
from hummingbot.core.iterators_threaded_timer import IteratorsThreadedTimer
from hummingbot.core.time_iterator_pp import TimeIteratorPurePython

log_records = []


class IteratorsThreadedTimerTest(unittest.TestCase):
    # logging.Level required to receive logs from the data source logger
    level = 0

    tick_size: float = 1
    data_dir: Path = root_path() / "test" / "profiling-data" / __file__.split('test/')[1].split('.')[0]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data_dir.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        super().setUp()

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
    def handle(record):
        global log_records
        log_records.append(record)

    @staticmethod
    def check_is_logged(log_level: str, message: str) -> bool:
        return any(
            record.levelname == log_level and record.getMessage() == message
            for record in log_records
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

    def test_init(self):
        list_iterators = []
        ticks = [300, 200, 200]
        priorities = [0, 1, 2]
        for ts, p in zip(ticks, priorities):
            time_iterator: TimeIteratorPurePython = TimeIteratorPurePython()
            time_iterator.tick_size = ts
            time_iterator.priority = p
            time_iterator.tick = lambda x: print(f"\tCalled with tick:{x}")
            list_iterators.append(time_iterator)

        timer = IteratorsThreadedTimer(200, list_iterators)

        self.assertNotEqual(list_iterators, list(timer.iterators))
        self.assertEqual(200, timer._period_s)
        self.assertEqual(200, timer.period_s)
        self.assertTrue(isinstance(timer.done, threading.Event))
        self.assertEqual([list_iterators[2], list_iterators[1]], list(timer.iterators))

    def test_iterators(self):
        list_iterators = []
        ticks = [300, 200, 200]
        priorities = [0, 1, 2]
        for ts, p in zip(ticks, priorities):
            time_iterator: TimeIteratorPurePython = TimeIteratorPurePython()
            time_iterator.tick_size = ts
            time_iterator.priority = p
            time_iterator.tick = lambda x: print(f"\tCalled with tick:{x}")
            list_iterators.append(time_iterator)

        timer = IteratorsThreadedTimer(200, list_iterators)

        self.assertEqual(timer._s_iterators, timer.iterators)

    def test__run_iterators(self):
        list_iterators = []
        ticks = [300, 200, 200]
        priorities = [0, 1, 2]
        for ts, p in zip(ticks, priorities):
            time_iterator: TimeIteratorPurePython = TimeIteratorPurePython()
            time_iterator.tick_size = ts
            time_iterator.priority = p
            list_iterators.append(time_iterator)
        timer = IteratorsThreadedTimer(200, list_iterators)
        with patch.object(TimeIteratorPurePython, 'tick') as mocked_tick:
            with patch.object(ClockPurePython, 'get_current_tick_s') as mocked_time:
                mocked_time.return_value = 1234567890
                timer._run_iterators()
        mocked_tick.assert_has_calls([call(1234567890), call(1234567890)])
        mocked_time.assert_has_calls([call(200), call(200)])

    def test_remove_iterator(self):
        list_iterators = []
        ticks = [300, 200, 200]
        priorities = [0, 1, 2]
        for ts, p in zip(ticks, priorities):
            time_iterator: TimeIteratorPurePython = TimeIteratorPurePython()
            time_iterator.tick_size = ts
            time_iterator.priority = p
            list_iterators.append(time_iterator)
        timer = IteratorsThreadedTimer(200, list_iterators)

        self.assertEqual([list_iterators[2], list_iterators[1]], list(timer.iterators))
        timer.remove_iterator(list_iterators[-1])
        self.assertEqual([list_iterators[-2]], list(timer.iterators))

    def test_add_iterator(self):
        list_iterators = []
        ticks = [300, 200, 200]
        priorities = [0, 1, 2]
        for ts, p in zip(ticks, priorities):
            time_iterator: TimeIteratorPurePython = TimeIteratorPurePython()
            time_iterator.tick_size = ts
            time_iterator.priority = p
            list_iterators.append(time_iterator)
        timer = IteratorsThreadedTimer(200, list_iterators)
        time_iterator: TimeIteratorPurePython = TimeIteratorPurePython()
        time_iterator.tick_size = 200
        time_iterator.priority = 3

        self.assertEqual([list_iterators[2], list_iterators[1]], list(timer.iterators))
        timer.add_iterator(time_iterator)
        self.assertEqual([time_iterator, list_iterators[2], list_iterators[1]], list(timer.iterators))

    def test_add_iterator_not_added(self):
        list_iterators = []
        ticks = [300, 200, 200]
        priorities = [0, 1, 2]
        for ts, p in zip(ticks, priorities):
            time_iterator: TimeIteratorPurePython = TimeIteratorPurePython()
            time_iterator.tick_size = ts
            time_iterator.priority = p
            list_iterators.append(time_iterator)
        timer = IteratorsThreadedTimer(200, list_iterators)

        # Trying to add iterator already in the list
        self.assertEqual([list_iterators[2], list_iterators[1]], list(timer.iterators))
        timer.add_iterator(list_iterators[2])
        self.assertEqual([list_iterators[2], list_iterators[1]], list(timer.iterators))

        # Trying to add iterator with wrong tick_size
        self.assertEqual([list_iterators[2], list_iterators[1]], list(timer.iterators))
        timer.add_iterator(list_iterators[0])
        self.assertEqual([list_iterators[2], list_iterators[1]], list(timer.iterators))

    def test_run(self):
        list_iterators = []
        ticks = [300, 200, 200]
        priorities = [0, 1, 2]
        for ts, p in zip(ticks, priorities):
            time_iterator: TimeIteratorPurePython = TimeIteratorPurePython()
            time_iterator.tick_size = ts
            time_iterator.priority = p
            time_iterator.tick = lambda x: print(f"\tCalled with tick:{x}")
            list_iterators.append(time_iterator)
        timer = IteratorsThreadedTimer(200, list_iterators)
        # Logger
        timer.logger().setLevel(1)
        timer.logger().addHandler(self)

        # Trying to add iterator already in the list
        self.assertEqual([list_iterators[2], list_iterators[1]], list(timer.iterators))
        with patch.object(timer.done, 'clear') as mocked_wait:
            with patch.object(timer.done, 'set') as mocked_set:
                with patch.object(time, 'time_ns') as mocked_time:
                    mocked_time.side_effect = [1, 3, 1, 3]
                    timer.run()
        mocked_wait.assert_called_once()
        mocked_set.assert_called_once()
        mocked_time.assert_has_calls([call(), call()])
        self.assertEqual(0, timer._drift_ns)
        self.assertEqual(2, timer._task_duration_ns)
        log_target = "Executing:\n"
        self.assertTrue(self.check_is_logged(log_level="INFO", message=log_target))

    def test_timer(self):
        list_iterators = []
        ticks = [3, 5, 5]
        priorities = [0, 1, 2]
        for ts, p in zip(ticks, priorities):
            time_iterator: TimeIteratorPurePython = TimeIteratorPurePython()
            time_iterator.tick_size = ts
            time_iterator.priority = p
            time_iterator.tick = lambda x: time.sleep(0.1)
            list_iterators.append(time_iterator)
        print("Initialized")
        timer = IteratorsThreadedTimer(0.5, list_iterators)

        timer.logger().setLevel(1)
        timer.logger().addHandler(self)

        print(f"Started at {time.time()}")
        timer.start()
        print("Waiting")
        time.sleep(4)
        timer.pause()
        # pprint(vars(timer))
        timer.unpause()
        time.sleep(4)
        timer.cancel()

        print(timer)
        print(log_records)
