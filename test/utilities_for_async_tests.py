"""
=========
utilities_for_async_tests
=========

The `utilities_for_async_tests` module provides some useful decorators to work with asynchronous functions in Python.

Description
-----------
The `utilities_for_async_tests` module contains the following decorators:

- `async_to_sync(func: Callable[..., Any]) -> Callable[..., Any]`: a decorator to convert an asynchronous function into
  a synchronous function.

- `async_run_with_concurrent_tasks(*long_running_functions: Callable[[], Coroutine[Any, Any, Any]]) -> Callable[[_F],
  _F]`: a decorator to run an asynchronous function with other long-running asynchronous functions concurrently.

- `async_gather_concurrent_tasks(*tasks: Callable[[], Coroutine[Any, Any, Any]]) -> Callable`: a decorator to run an
  asynchronous function and other asynchronous tasks concurrently.

Example usage
-------------
Here's an example usage of the `utilities_for_async_tests` module:

.. code-block:: python

    from utilities_for_async_tests import async_to_sync

    @async_to_sync
    async def async_func():
        await asyncio.sleep(1)
        return "Hello, World!"

    sync_result = async_func()
    print(sync_result)  # Output: Hello, World!

Module name: utilities_for_async_tests.py
Module description: A module that provides decorators to work with asynchronous functions in Python.
Copyright (c) 2023
License: MIT
Author: Unknown
Creation date: 2023/04/08
"""

import asyncio
import functools
from asyncio import Task
from typing import Any, Callable, Coroutine, List, TypeVar


def async_to_sync(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to convert an async function into a synchronous function.

    If there's an existing running loop, it uses `run_until_complete()` to execute the coroutine.
    Otherwise, it uses `asyncio.run()`.

    :param func: The async function to be converted.
    :type func: Callable[..., Any]
    :return: The wrapped synchronous function.
    :rtype: Callable[..., Any]

    Usage:

    .. code-block:: python

        from utilities_for_async_tests import async_to_sync_in_loop

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
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        else:
            # Create a new loop in a separate thread and run the coroutine there
            raise NotImplementedError("This decorator is not yet implemented to run in a running loop.")

        if loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                result = asyncio.run(func(*args, **kwargs))
                return result

        if loop.is_running():
            # Create a new loop in a separate thread and run the coroutine there
            raise NotImplementedError("This decorator is not yet implemented to run in a running loop.")
        else:
            try:
                result = loop.run_until_complete(func(*args, **kwargs))
            except RuntimeError:
                # It is possible that the current loop is closed. This will happen in a few situations:
                #   - The loop is closed by the IsolatedAsyncioTestCase class.
                #         This is normal, IsolatedAsyncioTestCase should be the tool to use for async tests
                #   - A side-effect of workaround of grabbing a new loop and closing it without resetting the main loop
                result = asyncio.run(func(*args, **kwargs))
                # loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
                # asyncio.set_event_loop(loop)
                # result = loop.run_until_complete(func(*args, **kwargs))
                # loop.close()

        return result

    return wrapper


def unwrap_function(f: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
    if hasattr(f, "__self__"):
        return getattr(f.__self__, f.__name__)
    elif hasattr(f, "__cls__"):
        return getattr(f.__cls__, f.__name__)
    return f


_F = TypeVar('_F', bound=Callable[..., Coroutine[Any, Any, Any]])


def async_run_with_concurrent_tasks(
        *tasks: Callable[[], Coroutine[Any, Any, Any]]
) -> Callable[[_F], _F]:
    """
    Decorator to run an async function with other long-running async functions concurrently.

    :param long_running_functions: Long-running async functions to run concurrently with the decorated function.
    :type long_running_functions: Callable[..., Awaitable[Any]]

    Usage:

    .. code-block:: python

        from utilities_for_async_tests import async_run_with_concurrent_tasks

        async def long_running_function():
            await asyncio.sleep(10)

        @async_run_with_concurrent_tasks(long_running_function)
        async def my_async_function():
            result = await some_operation()
            # ...
    """

    def decorator(coro: _F) -> _F:
        @functools.wraps(coro)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create tasks for all long-running functions
            long_running_tasks: List[Task] = [asyncio.create_task(unwrap_function(f)()) for f in tasks]
            result: Any = None
            try:
                # Run the original async test function
                result = await coro(*args, **kwargs)
            finally:
                # Cancel all long-running tasks once the other function is done
                for task in long_running_tasks:
                    task.cancel()

                for task in long_running_tasks:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            return result

        return wrapper

    return decorator


def async_gather_concurrent_tasks(*tasks: Callable[[], Coroutine[Any, Any, Any]]) -> Callable:
    """
    Decorator to run an async function with other long-running async functions concurrently.

    :param long_running_functions: Long-running async functions to run concurrently with the decorated function.
    :type long_running_functions: Callable[..., Awaitable[Any]]

    Usage:

    .. code-block:: python

        from utilities_for_async_tests import async_run_with_concurrent_tasks

        async def long_running_function():
            await asyncio.sleep(10)

        @async_gather_concurrent_tasks(long_running_function)
        async def my_async_function():
            result = await some_operation()
            # ...
    """

    def decorator(coro: _F) -> _F:
        @functools.wraps(coro)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async def _run_tasks() -> None:
                await asyncio.gather(*[unwrap_function(task)() for task in tasks], return_exceptions=False)

            results, _ = await asyncio.gather(coro(*args, **kwargs), _run_tasks(), return_exceptions=False)
            return results

        return wrapper

    return decorator
