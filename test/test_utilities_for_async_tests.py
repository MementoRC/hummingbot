import asyncio
import unittest
from test.utilities_for_async_tests import (
    async_gather_concurrent_tasks,
    async_run_with_concurrent_tasks,
    async_to_sync,
    unwrap_function,
)
from typing import Optional


class TestAsyncToSyncNoLoop(unittest.TestCase):
    @async_to_sync
    async def async_add(self, a: int, b: int) -> int:
        await asyncio.sleep(0.1)
        return a + b

    def test_async_add(self):
        result = self.async_add(1, 2)
        self.assertEqual(3, result)

    @async_to_sync
    async def async_raise_exception(self) -> None:
        await asyncio.sleep(0.1)
        raise ValueError("Test exception")

    def test_async_raise_exception(self):
        with self.assertRaises(ValueError) as context:
            self.async_raise_exception()
        self.assertEqual(str(context.exception), "Test exception")

    def test_async_to_sync_running_loop(self):
        # Test the behavior of the function when there's a running loop
        async def coro():
            with self.assertRaises(NotImplementedError):
                self.async_add(1, 2)

        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(coro())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(coro())
            loop.close()


class TestAsyncToSyncClassAttributeLoop(unittest.TestCase):
    _main_loop: Optional[asyncio.AbstractEventLoop]
    loop: asyncio.AbstractEventLoop

    @classmethod
    def setUpClass(cls):
        try:
            cls._main_loop = asyncio.get_event_loop()
        except RuntimeError:
            cls._main_loop = None
        cls.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls.loop)

    @classmethod
    def tearDownClass(cls):
        cls.loop.close()
        if cls._main_loop is not None:
            asyncio.set_event_loop(cls._main_loop)

    @async_to_sync
    async def async_add(self, a: int, b: int) -> int:
        await asyncio.sleep(0.1)
        return a + b

    def test_async_add(self):
        result = self.async_add(1, 2)
        self.assertEqual(3, result)


class TestAsyncToSyncRunningLoop(unittest.TestCase):
    def test_async_in_running_loop(self):
        async def _test():
            @async_to_sync
            async def async_add(a: int, b: int) -> int:
                await asyncio.sleep(0.1)
                return a + b

            def _test_sync():
                result = async_add(1, 2)
                self.assertEqual(3, result)

            # This will raise an exception since we are trying to decorate an async def
            # function into a sync context while the event loop is running.
            with self.assertRaises(NotImplementedError):
                _test_sync()

            await asyncio.sleep(0)

        asyncio.run(_test())


async def short_running_function() -> None:
    await asyncio.sleep(0.1)


async def long_running_function() -> None:
    await asyncio.sleep(2)


async def raising_exception_function() -> None:
    await asyncio.sleep(0.1)
    raise ValueError("Test exception")


class TestAsyncRunWithConcurrentTasks(unittest.TestCase):
    def test_single_concurrent_task(self):
        async def _test():
            @async_run_with_concurrent_tasks(short_running_function)
            async def async_add(a: int, b: int) -> int:
                await asyncio.sleep(0.1)
                return a + b

            result = await async_add(1, 2)
            self.assertEqual(3, result)

        asyncio.run(_test())

    def test_single_gather_concurrent_task(self):
        async def _test():
            @async_gather_concurrent_tasks(short_running_function)
            async def async_add(a: int, b: int) -> int:
                await asyncio.sleep(0.1)
                return a + b

            result = await async_add(1, 2)
            self.assertEqual(3, result)

        asyncio.run(_test())

    def test_multiple_concurrent_tasks(self):
        async def _test():
            @async_run_with_concurrent_tasks(short_running_function, long_running_function)
            async def async_add(a: int, b: int) -> int:
                await asyncio.sleep(0.1)
                return a + b

            result = await async_add(1, 2)
            self.assertEqual(3, result)

        asyncio.run(_test())

    def test_multiple_gather_concurrent_tasks(self):
        async def _test():
            @async_gather_concurrent_tasks(short_running_function, long_running_function)
            async def async_add(a: int, b: int) -> int:
                await asyncio.sleep(0.1)
                return a + b

            result = await async_add(1, 2)
            self.assertEqual(3, result)

        asyncio.run(_test())

    def test_concurrent_task_does_not_raise_exception(self):
        async def _test():
            @async_run_with_concurrent_tasks(raising_exception_function)
            async def async_add(a: int, b: int) -> int:
                await asyncio.sleep(0.1)
                return a + b

            await async_add(1, 2)

        asyncio.run(_test())

    def test_gather_concurrent_task_raises_exception(self):
        async def _test():
            @async_gather_concurrent_tasks(raising_exception_function)
            async def async_add(a: int, b: int) -> int:
                await asyncio.sleep(0.1)
                return a + b

            with self.assertRaises(ValueError) as context:
                await async_add(1, 2)
            self.assertEqual(str(context.exception), "Test exception")

        asyncio.run(_test())


class TestUnwrapFunction(unittest.TestCase):

    async def async_function(self) -> str:
        return "Instance method"

    @classmethod
    async def async_class_method(cls) -> str:
        return "Class method"

    @async_to_sync
    async def test_instance_method(self):
        instance = TestUnwrapFunction()
        bound_method = unwrap_function(instance.async_function)
        self.assertEqual(await bound_method(), "Instance method")

    @async_to_sync
    async def test_class_method(self):
        bound_method = unwrap_function(TestUnwrapFunction.async_class_method)
        self.assertEqual(await bound_method(), "Class method")

    @async_to_sync
    async def test_regular_function(self):
        async def async_regular_function() -> str:
            return "Regular function"

        function = unwrap_function(async_regular_function)
        self.assertEqual(await function(), "Regular function")


class TestAsyncRunWithConcurrentTasksWithUnwrapping(unittest.TestCase):
    async def instance_raising_exception_function(self) -> None:
        await asyncio.sleep(0.1)
        raise ValueError("Test exception")

    @classmethod
    async def class_raising_exception_function(cls) -> None:
        await asyncio.sleep(0.1)
        raise ValueError("Test exception")

    def test_gather_concurrent_task_raises_exception_instance(self):
        async def _test():
            @async_gather_concurrent_tasks(self.instance_raising_exception_function)
            async def async_add(a: int, b: int) -> int:
                await asyncio.sleep(0.1)
                return a + b

            with self.assertRaises(ValueError) as context:
                await async_add(1, 2)
            self.assertEqual(str(context.exception), "Test exception")

        asyncio.run(_test())

    def test_gather_concurrent_task_raises_exception_class(self):
        async def _test():
            @async_gather_concurrent_tasks(
                TestAsyncRunWithConcurrentTasksWithUnwrapping.class_raising_exception_function)
            async def async_add(a: int, b: int) -> int:
                await asyncio.sleep(0.1)
                return a + b

            with self.assertRaises(ValueError) as context:
                await async_add(1, 2)
            self.assertEqual(str(context.exception), "Test exception")

        asyncio.run(_test())


class TestAsyncCombinedDecorators(unittest.TestCase):

    @classmethod
    async def class_raising_exception_function(cls) -> None:
        await asyncio.sleep(0.1)
        raise ValueError("Test exception")

    async def instance_raising_exception_function(cls) -> None:
        await asyncio.sleep(0.1)
        raise ValueError("Test exception")

    async def async_add(self, a: int, b: int) -> int:
        await asyncio.sleep(0.1)
        return a + b

    @async_to_sync
    @async_run_with_concurrent_tasks(lambda: TestAsyncCombinedDecorators.class_raising_exception_function())
    async def test_async_add_with_both_decorators(self):
        result = await self.async_add(1, 2)
        self.assertEqual(3, result)

    # This will not work because the instance method is not bound to anything yet
    # @async_to_sync
    # @async_run_with_concurrent_tasks(functools.partial(instance_long_running_function, self))
    # async def test_async_add_with_both_decorators(self):
    #     result = await self.async_add(1, 2)
    #     self.assertEqual(3, result)


if __name__ == "__main__":
    unittest.main()
