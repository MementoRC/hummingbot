import asyncio
import concurrent.futures
import functools
import threading
import time
import unittest
from random import uniform
from time import sleep
from typing import Any, Callable, List, Optional, Tuple
from unittest.mock import Mock

from hummingbot.core.utils.async_event_thread_safe import (
    EventLoopProtocol,
    LockableProtocol,
    ThreadSafeAsyncioEvent,
    ThreadSafeAsyncioEventProtocol,
    ensure_thread_safety_coroutine,
    ensure_thread_safety_soon,
    lock_acquisition_decorator,
    loop_conditions_decorator,
)


def increment_counter_decorator(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Execute the method
        result = method(self, *args, **kwargs)
        # Increment the counter
        self.counter += 1
        return result

    return wrapper


class TestLockAcquisitionDecorator(unittest.TestCase):
    class TestLockableObject(LockableProtocol):
        def __init__(self):
            self.rnd = None
            self.counter = 0
            self.loop = 10
            self._lock = threading.Lock()
            self._async_lock = asyncio.Lock()

        def get_lock(self):
            return self._lock

        def get_async_lock(self):
            return self._async_lock

        @increment_counter_decorator
        def decrement_counter(self) -> int:
            self.counter = self.counter - 1
            return self.counter

        @lock_acquisition_decorator(use_lock=False)
        def adder_without_lock(self, amount: int, repeats: int):
            # python 3.10+ is less prone to race conditions
            for _ in range(repeats):
                tmp = self.counter
                # Suggest a context switch
                time.sleep(0)
                tmp = tmp + amount
                time.sleep(0)
                self.counter = tmp

        @lock_acquisition_decorator(use_lock=False)
        def subtractor_without_lock(self, amount: int, repeats: int):
            # python 3.10+ is less prone to race conditions
            for _ in range(repeats):
                tmp = self.counter
                time.sleep(0)
                tmp = tmp - amount
                time.sleep(0)
                self.counter = tmp

        @lock_acquisition_decorator(use_lock=True)
        def adder_with_lock(self, amount: int, repeats: int):
            # python 3.10+ is less prone to race conditions
            for _ in range(repeats):
                tmp = self.counter
                # Suggest a context switch
                time.sleep(0)
                tmp = tmp + amount
                time.sleep(0)
                self.counter = tmp

        @lock_acquisition_decorator(use_lock=True)
        def subtractor_with_lock(self, amount: int, repeats: int):
            # python 3.10+ is less prone to race conditions
            for _ in range(repeats):
                tmp = self.counter
                time.sleep(0)
                tmp = tmp - amount
                time.sleep(0)
                self.counter = tmp

        @lock_acquisition_decorator(use_lock=False)
        @increment_counter_decorator
        def adder_without_lock_decorated(self, amount: int, repeats: int):
            # python 3.10+ is less prone to race conditions
            for _ in range(repeats):
                tmp = self.counter
                # Suggest a context switch
                time.sleep(0)
                tmp = tmp + amount
                time.sleep(0)
                self.counter = tmp

        @lock_acquisition_decorator(use_lock=False)
        @increment_counter_decorator
        def subtractor_without_lock_decorated(self, amount: int, repeats: int):
            # python 3.10+ is less prone to race conditions
            for _ in range(repeats):
                tmp = self.counter
                time.sleep(0)
                tmp = tmp - amount
                time.sleep(0)
                self.counter = tmp

        @lock_acquisition_decorator(use_lock=True)
        @increment_counter_decorator
        def adder_with_lock_decorated(self, amount: int, repeats: int):
            # python 3.10+ is less prone to race conditions
            for _ in range(repeats):
                tmp = self.counter
                # Suggest a context switch
                time.sleep(0)
                tmp = tmp + amount
                time.sleep(0)
                self.counter = tmp

        @lock_acquisition_decorator(use_lock=True)
        @increment_counter_decorator
        def subtractor_with_lock_decorated(self, amount: int, repeats: int):
            # python 3.10+ is less prone to race conditions
            for _ in range(repeats):
                tmp = self.counter
                time.sleep(0)
                tmp = tmp - amount
                time.sleep(0)
                self.counter = tmp

        # --- Test of asyncio.Lock as locking mechanism

        @lock_acquisition_decorator(use_lock=False, locking_method="get_async_lock")
        async def async_adder_without_lock(self, amount: int, repeats: int):
            for _ in range(repeats):
                tmp = self.counter
                await asyncio.sleep(0)
                tmp = tmp + amount
                await asyncio.sleep(0)
                self.counter = tmp

        @lock_acquisition_decorator(use_lock=False, locking_method="get_async_lock")
        async def async_subtractor_without_lock(self, amount: int, repeats: int):
            for _ in range(repeats):
                tmp = self.counter
                await asyncio.sleep(0)
                tmp = tmp - amount
                await asyncio.sleep(0)
                self.counter = tmp

        @lock_acquisition_decorator(use_lock=True, locking_method="get_async_lock")
        async def async_adder_with_lock(self, amount: int, repeats: int):
            for _ in range(repeats):
                tmp = self.counter
                await asyncio.sleep(0)
                tmp = tmp + amount
                await asyncio.sleep(0)
                self.counter = tmp

        @lock_acquisition_decorator(use_lock=True, locking_method="get_async_lock")
        async def async_subtractor_with_lock(self, amount: int, repeats: int):
            for _ in range(repeats):
                tmp = self.counter
                await asyncio.sleep(0)
                tmp = tmp - amount
                await asyncio.sleep(0)
                self.counter = tmp

    def setUp(self) -> None:
        self.protocol = self.TestLockableObject()
        self.one_thread_increment = 10
        self.thread_count = 10

    def test_decrement_return_increment(self):
        # Demonstrate the behavior of the helping decorator
        # The method decrements the counter and returns the value
        self.assertEqual(-1, self.protocol.decrement_counter())
        # The decorator then increments the counter
        self.assertEqual(0, self.protocol.counter)

    def test_threads_with_race_condition(self):
        adder_thread = threading.Thread(target=self.protocol.adder_without_lock, args=(1, 1000))
        adder_thread.start()
        subtractor_thread = threading.Thread(target=self.protocol.subtractor_without_lock, args=(1, 1000))
        subtractor_thread.start()
        adder_thread.join()
        subtractor_thread.join()
        self.assertNotEqual(0, self.protocol.counter)

    def test_threads_with_race_condition_solved_with_lock(self):
        adder_thread = threading.Thread(target=self.protocol.adder_with_lock, args=(1, 1000))
        adder_thread.start()
        subtractor_thread = threading.Thread(target=self.protocol.subtractor_with_lock, args=(1, 1000))
        subtractor_thread.start()
        adder_thread.join()
        subtractor_thread.join()
        self.assertEqual(0, self.protocol.counter)

    def test_threads_with_race_condition_decorated(self):
        adder_thread = threading.Thread(target=self.protocol.adder_without_lock_decorated, args=(1, 1000))
        adder_thread.start()
        subtractor_thread = threading.Thread(target=self.protocol.subtractor_without_lock_decorated, args=(1, 1000))
        subtractor_thread.start()
        adder_thread.join()
        subtractor_thread.join()
        # The race condition prevents the counter from being 0
        self.assertNotEqual(0, self.protocol.counter)

    def test_threads_with_race_condition_solved_with_lock_decorated(self):
        adder_thread = threading.Thread(target=self.protocol.adder_with_lock_decorated, args=(1, 1000))
        adder_thread.start()
        subtractor_thread = threading.Thread(target=self.protocol.subtractor_with_lock_decorated, args=(1, 1000))
        subtractor_thread.start()
        adder_thread.join()
        subtractor_thread.join()

        # The increment_counter_decorator is called by the adder and the subtractor
        # Only the last thread of both adder and subtractor exercise the increment_counter_decorator
        self.assertEqual(2, self.protocol.counter)

    def test_async_threads_with_race_condition(self):
        async def run_test():
            tasks = [
                asyncio.create_task(self.protocol.async_adder_without_lock(1, 1000)),
                asyncio.create_task(self.protocol.async_subtractor_without_lock(1, 1000)),
            ]
            await asyncio.gather(*tasks)

        asyncio.run(run_test())
        self.assertNotEqual(0, self.protocol.counter)

    def test_async_threads_with_race_condition_solved_with_lock(self):
        async def run_test():
            tasks = [
                asyncio.create_task(self.protocol.async_adder_with_lock(1, 1000)),
                asyncio.create_task(self.protocol.async_subtractor_with_lock(1, 1000)),
            ]
            await asyncio.gather(*tasks)

        asyncio.run(run_test())
        self.assertEqual(0, self.protocol.counter)


class TestLoopConditionsDecorator(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.protocol = Mock(spec=EventLoopProtocol)
        self.protocol.get_loop.return_value = self.loop

    def tearDown(self) -> None:
        self.loop.close()

    def test_same_loop(self):
        @loop_conditions_decorator
        def same_loop_function(self: Any, *args, **kwargs) -> str:
            return "success"

        async def run_same_loop_function():
            self.assertTrue(self.loop.is_running())
            return same_loop_function(self.protocol)

        result = self.loop.run_until_complete(run_same_loop_function())
        self.assertEqual(result, "success")

    def test_loop_arguments_passed(self):
        @loop_conditions_decorator
        def loop_arguments_passed_function(
                self: Any,
                *args,
                creation_loop=None, current_loop=None) -> Tuple[bool, bool]:
            is_creation_loop_set = creation_loop is not None
            is_current_loop_set = current_loop is not None
            return is_creation_loop_set, is_current_loop_set

        async def run_loop_arguments_passed_function():
            self.assertTrue(self.loop.is_running())
            return loop_arguments_passed_function(self.protocol)

        result = self.loop.run_until_complete(run_loop_arguments_passed_function())
        self.assertEqual(result, (True, True))

    def test_different_loop(self):
        async def simulate_coroutine():
            async def run_in_different_loop():
                @loop_conditions_decorator
                def different_loop_function(self: Any, *args, **kwargs) -> str:
                    return "success"

                # Main event loop is not running: Exception should be raised
                self.assertFalse(self.loop.is_running())
                with self.assertRaises(RuntimeError):
                    different_loop_function(self.protocol)

            # Main event loop is running: Function should be executed
            # in a different thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                new_loop = executor.submit(asyncio.new_event_loop).result()
                future = executor.submit(new_loop.run_until_complete, run_in_different_loop())
                result = future.result()
                exception = future.exception()
                executor.submit(new_loop.close)
                if exception:
                    raise exception
                return result

        asyncio.run(simulate_coroutine())

    def test_different_loop_main_loop_not_running(self):
        async def run_in_different_loop():
            @loop_conditions_decorator
            def different_loop_function(self: Any, *args, **kwargs) -> str:
                return "success"

            # Main event loop is not running: Exception should be raised
            self.assertFalse(self.loop.is_running())
            with self.assertRaises(RuntimeError):
                different_loop_function(self.protocol)

        new_loop = asyncio.new_event_loop()
        try:
            new_loop.run_until_complete(run_in_different_loop())
        finally:
            new_loop.close()

    def test_closed_loop(self):
        self.loop.close()

        @loop_conditions_decorator
        def closed_loop_function(self: Any, *args, **kwargs) -> str:
            return "success"

        # Main event loop is closed: Exception should be raised
        self.assertTrue(self.loop.is_closed())
        with self.assertRaises(RuntimeError):
            closed_loop_function(self.protocol)

    def test_no_running_loop(self):
        @loop_conditions_decorator
        def no_running_loop_function(self: Any, *args, **kwargs) -> str:
            with self.assertRaises(RuntimeError):
                asyncio.get_running_loop()
            return "success"

        with self.assertRaises(RuntimeError):
            no_running_loop_function(self.protocol)


class TestEnsureThreadSafetyDecorator(unittest.TestCase):
    class TestClass(ThreadSafeAsyncioEventProtocol):
        def __init__(self, loop: asyncio.AbstractEventLoop):
            if loop is None or loop.is_closed():
                raise ValueError("loop must be a running event loop")
            self._lock = threading.Lock()
            self._loop = loop
            self._current_loop: Optional[asyncio.AbstractEventLoop] = None
            self.concurrent_executions: int = 0

        def get_loop(self):
            return self._loop

        def get_lock(self):
            return self._lock

        def called_on_init_loop(self):
            return self._called_on_init_loop

        def set_called_on_init_loop(self, value: bool):
            self._called_on_init_loop = value

        @ensure_thread_safety_soon(use_lock=False)
        def test_method(self) -> str:
            return "test_method"

        @ensure_thread_safety_soon(use_lock=True)
        def test_method_with_lock(self) -> str:
            return "test_method_with_lock"

        @ensure_thread_safety_soon(use_lock=False)
        def test_wait(self, time: float, offset: float = 0.001) -> Tuple[float, int]:
            self.concurrent_executions = self.concurrent_executions + 1
            if self._lock.locked():
                self.concurrent_executions = 0
            return time + offset, self.concurrent_executions

        @ensure_thread_safety_soon(use_lock=True)
        def test_wait_with_lock(self, time: float, offset: float = 0.001) -> Tuple[float, int]:
            self.concurrent_executions = self.concurrent_executions + 1
            if self._lock.locked():
                self.concurrent_executions = 0
            return time + offset, self.concurrent_executions

    def test_decorated_method_same_loop(self):
        async def run_test():
            loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            self.assertFalse(loop.is_closed())
            self.assertTrue(loop.is_running())
            test_obj = self.TestClass(loop=loop)
            self.assertEqual("test_method", test_obj.test_method())
            self.assertTrue(test_obj.called_on_init_loop())

        asyncio.run(run_test())

    def test_decorated_method_same_loop_with_lock(self):
        async def run_test():
            loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            self.assertFalse(loop.is_closed())
            self.assertTrue(loop.is_running())
            test_obj = self.TestClass(loop=loop)
            self.assertEqual("test_method_with_lock", test_obj.test_method_with_lock())
            self.assertTrue(test_obj.called_on_init_loop())

        asyncio.run(run_test())

    def test_decorated_method_different_loop(self):
        async def run_test():
            primary_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            self.assertFalse(primary_loop.is_closed())
            self.assertTrue(primary_loop.is_running())

            # Starting a new thread to run the test method on its own loop
            asserts: List = []
            test_obj = self.TestClass(loop=primary_loop)
            self.assertTrue(primary_loop == test_obj.get_loop())

            def event_loop_in_thread(test_obj_on_primary, results):
                thread_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
                asyncio.set_event_loop(thread_loop)

                results.append(thread_loop is not primary_loop)
                results.append(thread_loop is not test_obj.get_loop())

                async def run_thread_test():
                    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
                    results.append(not loop.is_closed())
                    results.append(loop.is_running())
                    results.append(test_obj_on_primary.test_method() == "test_method")
                    results.append(not test_obj.called_on_init_loop())

                asyncio.run(run_thread_test())

            thread = threading.Thread(target=event_loop_in_thread, args=(test_obj, asserts))
            thread.start()
            # We need to await to release the primary_loop to allow a response to the
            # test_method call from the thread
            await asyncio.sleep(0.5)
            thread.join()

            self.assertTrue(all(asserts))

        asyncio.run(run_test())

    def test_decorated_method_different_loop_with_lock(self):
        async def run_test():
            primary_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            self.assertFalse(primary_loop.is_closed())
            self.assertTrue(primary_loop.is_running())

            # Starting a new thread to run the test method on its own loop
            asserts: List = []
            test_obj = self.TestClass(loop=primary_loop)
            self.assertTrue(primary_loop == test_obj.get_loop())

            def event_loop_in_thread(test_obj_on_primary, results):
                thread_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
                asyncio.set_event_loop(thread_loop)

                results.append(thread_loop is not primary_loop)
                results.append(thread_loop is not test_obj.get_loop())

                async def run_thread_test():
                    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
                    results.append(not loop.is_closed())
                    results.append(loop.is_running())
                    results.append(test_obj_on_primary.test_method_with_lock() == "test_method_with_lock")
                    results.append(not test_obj.called_on_init_loop())

                asyncio.run(run_thread_test())

            thread = threading.Thread(target=event_loop_in_thread, args=(test_obj, asserts))
            thread.start()
            # We need to await to release the primary_loop to allow a response to the
            # test_method call from the thread
            await asyncio.sleep(0.5)
            thread.join()

            self.assertTrue(all(asserts))

        asyncio.run(run_test())

    def test_decorated_method_different_thread(self):
        primary_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(primary_loop)

        def run_test():
            # Wait for the primary loop to start since it is initiated after the thread started
            sleep(0.1)
            test_obj = self.TestClass(loop=primary_loop)
            self.assertTrue(primary_loop == test_obj.get_loop())
            assert primary_loop.is_running()

            # However, there is no running loop in the thread, so RuntimeError is raised
            with self.assertRaises(RuntimeError):
                asyncio.get_running_loop()

            # This should also raise an exception because no loop is running in the thread
            with self.assertRaises(RuntimeError):
                test_obj.test_method()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_test)
            primary_loop.run_until_complete(asyncio.sleep(0.5))
            result = future.result()
            self.assertEqual(result, None)
        # Remember to close the loop
        primary_loop.close()

    def test_decorated_method_different_thread_with_lock(self):
        primary_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(primary_loop)

        def run_test():
            # Wait for the primary loop to start since it is initiated after the thread started
            sleep(0.1)
            test_obj = self.TestClass(loop=primary_loop)
            self.assertTrue(primary_loop == test_obj.get_loop())
            assert primary_loop.is_running()

            # However, there is no running loop in the thread, so RuntimeError is raised
            with self.assertRaises(RuntimeError):
                asyncio.get_running_loop()

            # This should also raise an exception because no loop is running in the thread
            with self.assertRaises(RuntimeError):
                test_obj.test_method_with_lock()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_test)
            primary_loop.run_until_complete(asyncio.sleep(0.5))
            result = future.result()
            self.assertEqual(result, None)
        # Remember to close the loop
        primary_loop.close()

    def test_main_loop_not_running_or_closed(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.stop()
        test_obj = self.TestClass(loop)

        with self.assertRaises(RuntimeError):
            test_obj.test_method()

        loop.close()

    def test_concurrent_calls(self):
        async def run_test():
            primary_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            self.assertFalse(primary_loop.is_closed())
            self.assertTrue(primary_loop.is_running())

            test_obj = self.TestClass(loop=primary_loop)

            def test_wait(t, offset) -> Tuple[float, int]:
                async def run():
                    # This is critical to prevent the loop from being blocked
                    await asyncio.sleep(0)
                    time, calls = test_obj.test_wait(t, offset=offset)
                    return time, calls

                return asyncio.run(run())

            async def schedule_test_wait(time, offset: float = 0) -> Tuple[float, int]:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = asyncio.wrap_future(executor.submit(test_wait, time, offset))
                    return await future

            async def gather_tasks():
                offset = 0.001
                input = [uniform(0, 1) for _ in range(10)]

                tasks = [schedule_test_wait(i, offset=offset) for i in input]
                results = await asyncio.gather(*tasks)

                self.assertTrue(len(results) == len(input))
                self.assertTrue(all([f"{i + offset:.5e}" == f"{r[0]:.5e}" for i, r in zip(input, results)]))
                self.assertFalse(all([r[1] == 0 for r in results]))

            await gather_tasks()

        asyncio.run(run_test())

    def test_concurrent_calls_with_lock(self):
        async def run_test():
            primary_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            self.assertFalse(primary_loop.is_closed())
            self.assertTrue(primary_loop.is_running())

            test_obj = self.TestClass(loop=primary_loop)

            def test_wait(t, offset) -> Tuple[float, int]:
                async def run():
                    # This is critical to prevent the loop from being blocked
                    await asyncio.sleep(0)
                    time, calls = test_obj.test_wait_with_lock(t, offset=offset)
                    return time, calls

                return asyncio.run(run())

            async def schedule_test_wait(time, offset: float = 0) -> Tuple[float, int]:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = asyncio.wrap_future(executor.submit(test_wait, time, offset))
                    return await future

            async def gather_tasks():
                offset = 0.001
                input = [uniform(0, 1) for _ in range(10)]

                tasks = [schedule_test_wait(i, offset=offset) for i in input]
                results = await asyncio.gather(*tasks)

                self.assertTrue(len(results) == len(input))
                self.assertTrue(all([f"{i + offset:.5e}" == f"{r[0]:.5e}" for i, r in zip(input, results)]))
                self.assertTrue(all([r[1] == 0 for r in results]))

            await gather_tasks()

        asyncio.run(run_test())


class TestEnsureThreadSafetyCoroutineDecorator(unittest.TestCase):
    class TestClass(ThreadSafeAsyncioEventProtocol):
        def __init__(self, loop: asyncio.AbstractEventLoop):
            if loop is None or loop.is_closed():
                raise ValueError("loop must be a running event loop")
            self._lock = threading.Lock()
            self._loop = loop
            self._current_loop: Optional[asyncio.AbstractEventLoop] = None
            self.concurrent_executions: int = 0

        def get_loop(self):
            return self._loop

        def get_lock(self):
            return self._lock

        def called_on_init_loop(self):
            return self._called_on_init_loop

        def set_called_on_init_loop(self, value: bool):
            self._called_on_init_loop = value

        @ensure_thread_safety_coroutine(use_lock=False)
        async def test_method(self) -> str:
            return "test_method"

        @ensure_thread_safety_coroutine(use_lock=True)
        async def test_method_with_lock(self) -> str:
            return "test_method_with_lock"

        @ensure_thread_safety_coroutine(use_lock=False)
        async def test_wait(self, time: float, offset: float = 0.001) -> Tuple[float, int]:
            self.concurrent_executions = self.concurrent_executions + 1
            if self._lock.locked():
                self.concurrent_executions = 0
            await asyncio.sleep(time)
            return time + offset, self.concurrent_executions

        @ensure_thread_safety_coroutine(use_lock=True)
        async def test_wait_with_lock(self, time: float, offset: float = 0.001) -> Tuple[float, int]:
            self.concurrent_executions = self.concurrent_executions + 1
            if self._lock.locked():
                self.concurrent_executions = 0
            await asyncio.sleep(time)
            return time + offset, self.concurrent_executions

    def test_decorated_method_same_loop(self):
        async def run_test():
            loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            self.assertFalse(loop.is_closed())
            self.assertTrue(loop.is_running())
            test_obj = self.TestClass(loop=loop)
            self.assertEqual("test_method", await test_obj.test_method())
            self.assertTrue(test_obj.called_on_init_loop())

        asyncio.run(run_test())

    def test_decorated_method_same_loop_with_lock(self):
        async def run_test():
            loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            self.assertFalse(loop.is_closed())
            self.assertTrue(loop.is_running())
            test_obj = self.TestClass(loop=loop)
            self.assertEqual("test_method_with_lock", await test_obj.test_method_with_lock())
            self.assertTrue(test_obj.called_on_init_loop())

        asyncio.run(run_test())

    def test_decorated_method_different_loop(self):
        async def run_test():
            primary_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            self.assertFalse(primary_loop.is_closed())
            self.assertTrue(primary_loop.is_running())

            # Starting a new thread to run the test method on its own loop
            asserts: List = []
            test_obj = self.TestClass(loop=primary_loop)
            self.assertTrue(primary_loop == test_obj.get_loop())

            def event_loop_in_thread(test_obj_on_primary, results):
                thread_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
                asyncio.set_event_loop(thread_loop)

                results.append(thread_loop is not primary_loop)
                results.append(thread_loop is not test_obj.get_loop())

                async def run_thread_test():
                    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
                    results.append(not loop.is_closed())
                    results.append(loop.is_running())
                    results.append(await test_obj_on_primary.test_method() == "test_method")
                    results.append(not test_obj.called_on_init_loop())

                asyncio.run(run_thread_test())

            thread = threading.Thread(target=event_loop_in_thread, args=(test_obj, asserts))
            thread.start()
            # We need to await to release the primary_loop to allow a response to the
            # test_method call from the thread
            await asyncio.sleep(0.5)
            thread.join()

            self.assertTrue(all(asserts))

        asyncio.run(run_test())

    def test_decorated_method_different_loop_with_lock(self):
        async def run_test():
            primary_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            self.assertFalse(primary_loop.is_closed())
            self.assertTrue(primary_loop.is_running())

            # Starting a new thread to run the test method on its own loop
            asserts: List = []
            test_obj = self.TestClass(loop=primary_loop)
            self.assertTrue(primary_loop == test_obj.get_loop())

            def event_loop_in_thread(test_obj_on_primary, results):
                thread_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
                asyncio.set_event_loop(thread_loop)

                results.append(thread_loop is not primary_loop)
                results.append(thread_loop is not test_obj.get_loop())

                async def run_thread_test():
                    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
                    results.append(not loop.is_closed())
                    results.append(loop.is_running())
                    results.append(await test_obj_on_primary.test_method_with_lock() == "test_method_with_lock")
                    results.append(not test_obj.called_on_init_loop())

                asyncio.run(run_thread_test())

            thread = threading.Thread(target=event_loop_in_thread, args=(test_obj, asserts))
            thread.start()
            # We need to await to release the primary_loop to allow a response to the
            # test_method call from the thread
            await asyncio.sleep(0.5)
            thread.join()

            self.assertTrue(all(asserts))

        asyncio.run(run_test())

    def test_decorated_method_different_thread(self):
        primary_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(primary_loop)

        async def run_test():
            # Wait for the primary loop to start since it is initiated after the thread started
            await asyncio.sleep(0.1)
            test_obj = self.TestClass(loop=primary_loop)
            self.assertTrue(primary_loop == test_obj.get_loop())
            assert primary_loop.is_running()

            # However, there is no running loop in the thread, so RuntimeError is raised
            with self.assertRaises(RuntimeError):
                asyncio.get_running_loop()

            # This should also raise an exception because no loop is running in the thread
            with self.assertRaises(RuntimeError):
                await test_obj.test_method()

        try:
            # Run the test in a new thread
            asyncio.to_thread(run_test)
        finally:
            # Remember to close the loop
            primary_loop.close()

    def test_decorated_method_different_thread_with_lock(self):
        primary_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(primary_loop)

        async def run_test():
            # Wait for the primary loop to start since it is initiated after the thread started
            await asyncio.sleep(0.1)
            test_obj = self.TestClass(loop=primary_loop)
            self.assertTrue(primary_loop == test_obj.get_loop())
            assert primary_loop.is_running()

            # However, there is no running loop in the thread, so RuntimeError is raised
            with self.assertRaises(RuntimeError):
                asyncio.get_running_loop()

            # This should also raise an exception because no loop is running in the thread
            with self.assertRaises(RuntimeError):
                await test_obj.test_method_with_lock()

        try:
            # Run the test in a new thread
            asyncio.to_thread(run_test)
        finally:
            # Remember to close the loop
            primary_loop.close()

    def test_main_loop_not_running_or_closed(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.stop()
        test_obj = self.TestClass(loop)

        with self.assertRaises(RuntimeError):
            loop.run_until_complete(test_obj.test_method())

        loop.close()

    def test_concurrent_calls(self):
        async def run_test():
            primary_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            self.assertFalse(primary_loop.is_closed())
            self.assertTrue(primary_loop.is_running())

            test_obj = self.TestClass(loop=primary_loop)

            def test_wait(t, offset) -> Tuple[float, int]:
                async def run():
                    # This is critical to prevent the loop from being blocked
                    await asyncio.sleep(0)
                    time, calls = await test_obj.test_wait(t, offset=offset)
                    return time, calls

                return asyncio.run(run())

            async def schedule_test_wait(time, offset: float = 0) -> Tuple[float, int]:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = asyncio.wrap_future(executor.submit(test_wait, time, offset))
                    return await future

            async def gather_tasks():
                offset = 0.001
                input = [uniform(0, 1) for _ in range(10)]

                tasks = [schedule_test_wait(i, offset=offset) for i in input]
                results = await asyncio.gather(*tasks)

                self.assertTrue(len(results) == len(input))
                self.assertTrue(all([f"{i + offset:.5e}" == f"{r[0]:.5e}" for i, r in zip(input, results)]))
                self.assertFalse(all([r[1] == 0 for r in results]))

            await gather_tasks()

        asyncio.run(run_test())

    def test_concurrent_calls_with_lock(self):
        async def run_test():
            primary_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
            self.assertFalse(primary_loop.is_closed())
            self.assertTrue(primary_loop.is_running())

            test_obj = self.TestClass(loop=primary_loop)

            def test_wait(t, offset) -> Tuple[float, int]:
                async def run():
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    time, calls = await test_obj.test_wait_with_lock(t, offset=offset)
                    return time, calls

                return asyncio.run(run())

            async def schedule_test_wait(time, offset: float = 0) -> Tuple[float, int]:
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    future = asyncio.wrap_future(executor.submit(test_wait, time, offset))
                    return await future

            async def gather_tasks():
                offset = 0.001
                input = [uniform(0, 1) for _ in range(10)]
                tasks = [schedule_test_wait(i, offset) for i in input]
                results = await asyncio.gather(*tasks)

                print(input, results)

                self.assertTrue(len(results) == len(input))
                self.assertTrue(all([f"{i + offset:.5e}" == f"{r[0]:.5e}" for i, r in zip(input, results)]))
                self.assertTrue(all([r[1] == 0 for r in results]))

            await gather_tasks()

        asyncio.run(run_test())


class TestThreadSafeAsyncioEvent(unittest.TestCase):
    @staticmethod
    def run_in_thread(func: Callable[..., Any], *args, **kwargs) -> Any:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(func(*args, **kwargs))

    def test_set_clear_is_set(self):
        async def test_coroutine():
            tr_event = asyncio.Event()
            self.assertFalse(tr_event.is_set())
            ts_event = ThreadSafeAsyncioEvent()
            self.assertFalse(ts_event.is_set())

            tr_event.set()
            self.assertTrue(tr_event.is_set())
            ts_event.set()
            self.assertTrue(ts_event.is_set())

            tr_event.clear()
            self.assertFalse(tr_event.is_set())
            ts_event.clear()
            self.assertFalse(ts_event.is_set())

        asyncio.run(test_coroutine())

    def test_cross_thread_events(self):
        async def setter(ts_event: ThreadSafeAsyncioEvent):
            await asyncio.sleep(1)
            ts_event.set()

        async def waiter(ts_event: ThreadSafeAsyncioEvent):
            await ts_event.wait()

        async def main_test_coroutine():
            ts_event = ThreadSafeAsyncioEvent()
            self.assertFalse(ts_event.is_set())

            with concurrent.futures.ThreadPoolExecutor() as executor:
                await asyncio.gather(
                    asyncio.get_event_loop().run_in_executor(executor, self.run_in_thread, setter, ts_event),
                    asyncio.get_event_loop().run_in_executor(executor, self.run_in_thread, waiter, ts_event),
                )

            self.assertTrue(ts_event.is_set())

        asyncio.run(main_test_coroutine())

    def test_multiple_events(self):
        async def setter(ts_event: ThreadSafeAsyncioEvent):
            await asyncio.sleep(1)
            ts_event.set()

        async def waiter(ts_event: ThreadSafeAsyncioEvent):
            await ts_event._wait()

        async def main_test_coroutine():
            ts_events = [ThreadSafeAsyncioEvent() for _ in range(3)]

            for ts_event in ts_events:
                self.assertFalse(ts_event.is_set())

            with concurrent.futures.ThreadPoolExecutor() as executor:
                await asyncio.gather(
                    *[asyncio.get_event_loop().run_in_executor(executor, self.run_in_thread, setter, ts_event) for
                      ts_event in ts_events],
                    *[asyncio.get_event_loop().run_in_executor(executor, self.run_in_thread, waiter, ts_event) for
                      ts_event in ts_events],
                )

            for ts_event in ts_events:
                self.assertTrue(ts_event.is_set())

        asyncio.run(main_test_coroutine())
