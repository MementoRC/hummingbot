import asyncio
import functools
import sys
import unittest
from asyncio import Task
from collections.abc import Set
from typing import Any, Awaitable, Callable, Coroutine, TypeVar

# When pytest with asyncio_mode=auto drives the test suite, it already provides
# proper event-loop isolation.  The manual save/restore/assert dance in the
# wrapper classes was needed for the bare-unittest runner; under pytest it
# actively fights the framework and causes spurious failures.
_PYTEST_RUNNER = "pytest" in sys.modules

T = TypeVar("T")


def async_to_sync(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """
    Decorator to convert an async function into a function that can be called in sync context.

    If there's an existing running loop, it uses `run_until_complete()` to execute the coroutine.
    Otherwise, it uses `asyncio.run()`.

    :param func: The async function to be converted.
    :type func: Callable[..., Any]
    :return: The wrapped synchronous function.
    :rtype: Callable[..., Any]

    Usage:

    .. code-block:: python

        from my_decorators import async_to_sync_in_loop

        class MyClass:
            @async_to_sync_in_loop
            async def async_method(self) -> str:
                await asyncio.sleep(1)
                return "Hello, World!"

        my_instance = MyClass()
        result = my_instance.async_method()
        print(result)  # Output: Hello, World!
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        except RuntimeError:
            return asyncio.run(func(*args, **kwargs))

        result: T = loop.run_until_complete(func(*args, **kwargs))
        return result

    return wrapper


class IsolatedAsyncioWrapperTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Custom test case class that wraps `unittest.IsolatedAsyncioTestCase`.

    Saves and restores the "main" event loop around each test so that one
    test's loop destruction cannot cascade into subsequent tests.

    Under pytest with ``asyncio_mode = auto``, the loop save/restore is
    still needed (other tests may kill the loop), but the identity
    assertions are relaxed since pytest-asyncio may reuse the same loop
    object.
    """

    main_event_loop = None

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.main_event_loop = asyncio.get_event_loop()
        except RuntimeError:
            cls.main_event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cls.main_event_loop)
        assert cls.main_event_loop is not None
        super().setUpClass()

    def setUp(self) -> None:
        self.local_event_loop = asyncio.get_event_loop()
        # Under pytest-asyncio the local and main loop may be the same object;
        # only assert they differ when running under the bare-unittest runner.
        if not _PYTEST_RUNNER:
            assert self.local_event_loop is not self.main_event_loop
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()
        if self.main_event_loop is not None and not self.main_event_loop.is_closed():
            asyncio.set_event_loop(self.main_event_loop)
            if not _PYTEST_RUNNER:
                assert asyncio.get_event_loop() is self.main_event_loop
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        if cls.main_event_loop is not None and not cls.main_event_loop.is_closed():
            asyncio.set_event_loop(cls.main_event_loop)
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())
        assert asyncio.get_event_loop() is not None

    def run_async_with_timeout(self, coroutine: Awaitable, timeout: float = 1.0) -> Any:
        """
        Run the given coroutine with a timeout.

        :param Awaitable coroutine: The coroutine to be executed.
        :param float timeout: The timeout value in seconds.
        :return: The result of the coroutine.
        :rtype: Any
        """
        return self.local_event_loop.run_until_complete(asyncio.wait_for(coroutine, timeout=timeout))

    @staticmethod
    async def await_task_completion(tasks_name: str | list[str] | None) -> None:
        """
        Await the completion of the given task.

        Warning: This method relies on undocumented method of Task (get_coro()),
        as well as internals of Python coroutines (cr_code.co_name).

        :param str tasks_name: The task name (or names) to be awaited.
        :return: The result of the task.
        """

        def get_coro_func_name(task):
            coro = task.get_coro()
            return coro.cr_code.co_name

        if tasks_name is None:
            return
        if isinstance(tasks_name, str):
            tasks_name = [tasks_name]
        tasks: Set[Task] = asyncio.all_tasks()
        tasks = {task for task in tasks for task_name in tasks_name if task_name == get_coro_func_name(task)}

        if tasks:
            await asyncio.wait(tasks)


class LocalClassEventLoopWrapperTestCase(unittest.TestCase):
    """
    Custom test case class that wraps `unittest.TestCase`.

    This class provides additional functionality to manage the main event loop and a local event loop for tests.
    It ensures that each test case runs in a local asyncio event loop, preventing interference between test suites.

    Example usage:
    ```python
    class MyTestCase(LocalClassEventLoopWrapperTestCase):
        def test_my_async_function(self):
            self.local_event_loop.run_until_complete(asyncio.sleep(0.1))
            ...
    ```

    Note:
    - It is important to make sure that all async functions in the test case are prefixed with the `async` keyword.
    - This class assumes that the tests are defined as methods in a subclass.

    Attributes:
    - `main_event_loop`: The reference to the main asyncio event loop.
    - `local_event_loop`: The local asyncio event loop used for each test case.
    """

    main_event_loop: asyncio.AbstractEventLoop | None = None
    local_event_loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            cls.main_event_loop = asyncio.get_event_loop()
        except RuntimeError:
            cls.main_event_loop = asyncio.new_event_loop()

        cls.local_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls.local_event_loop)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.local_event_loop is not None:
            tasks: Set[Task] = asyncio.all_tasks(cls.local_event_loop)
            for task in tasks:
                task.cancel()
            cls.local_event_loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            cls.local_event_loop.run_until_complete(cls.local_event_loop.shutdown_asyncgens())
            cls.local_event_loop.close()
            cls.local_event_loop = None

        asyncio.set_event_loop(cls.main_event_loop)
        cls.main_event_loop = None
        super().tearDownClass()

    def run_async_with_timeout(self, coroutine: Awaitable, timeout: float = 1.0) -> Any:
        """
        Run the given coroutine with a timeout.

        :param Awaitable coroutine: The coroutine to be executed.
        :param float timeout: The timeout value in seconds.
        :return: The result of the coroutine.
        :rtype: Any
        """
        return self.local_event_loop.run_until_complete(asyncio.wait_for(coroutine, timeout=timeout))


class LocalTestEventLoopWrapperTestCase(unittest.TestCase):
    """
    Custom test case class that wraps `unittest.TestCase`.

    This class provides additional functionality to manage the main event loop and a local event loop for each test.
    It ensures that each test case runs in a local asyncio event loop, preventing interference between test suites.

    Example usage:
    ```python
    class MyTestCase(LocalTestEventLoopWrapperTestCase):
        def test_my_async_function(self):
            self.local_event_loop.run_until_complete(asyncio.sleep(0.1))
            ...
    ```

    Note:
    - It is important to make sure that all async functions in the test case are prefixed with the `async` keyword.
    - This class assumes that the tests are defined as methods in a subclass.

    Attributes:
    - `main_event_loop`: The reference to the main asyncio event loop.
    - `local_event_loop`: The local asyncio event loop used for each test case.
    """

    main_event_loop: asyncio.AbstractEventLoop | None = None
    local_event_loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            cls.main_event_loop = asyncio.get_event_loop()
        except RuntimeError:
            cls.main_event_loop = None

    def setUp(self) -> None:
        super().setUp()
        self.local_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.local_event_loop)
        self.assertEqual(asyncio.get_event_loop(), self.local_event_loop)

    def tearDown(self) -> None:
        if self.local_event_loop is not None:
            tasks: Set[Task] = asyncio.all_tasks(self.local_event_loop)
            for task in tasks:
                task.cancel()
            self.local_event_loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            self.local_event_loop.run_until_complete(self.local_event_loop.shutdown_asyncgens())
            self.local_event_loop.close()
            self.local_event_loop = None
        super().tearDown()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.main_event_loop is not None and not cls.main_event_loop.is_closed():
            asyncio.set_event_loop(cls.main_event_loop)
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())

        cls.main_event_loop = None
        super().tearDownClass()

    def run_async_with_timeout(self, coroutine: Awaitable, timeout: float = 1.0) -> Any:
        """
        Run the given coroutine with a timeout.

        :param Awaitable coroutine: The coroutine to be executed.
        :param float timeout: The timeout value in seconds.
        :return: The result of the coroutine.
        :rtype: Any
        """
        return self.local_event_loop.run_until_complete(asyncio.wait_for(coroutine, timeout=timeout))
