"""
A module that provides a thread-safe version of the asyncio.Event class,
which supports communication across multiple event loops running in separate threads.

This module includes decorators to ensure thread safety for both synchronous
and asynchronous functions. It also provides a class called ThreadSafeAsyncioEvent,
which encapsulates the functionality of the asyncio.Event class with additional
thread safety and cross-event-loop communication support using the
`asyncio.run_coroutine_threadsafe()` and `call_soon_threadsafe` methods.

Example usage:

    async def main():
        ts_event = ThreadSafeAsyncioEvent()

        async def setter():
            await asyncio.sleep(1)
            ts_event.set()
            print("Event set")

        async def waiter():
            print("Waiting for event")
            ts_event.wait()
            print("Event received")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            await asyncio.gather(
                asyncio.get_event_loop().run_in_executor(executor, run_in_thread, setter),
                asyncio.get_event_loop().run_in_executor(executor, run_in_thread, waiter),
            )

    if __name__ == "__main__":
        asyncio.run(main())
"""

import asyncio
import concurrent.futures
import functools
import threading
from typing import Any, Awaitable, Callable, Optional, Protocol, TypeVar, Union

T = TypeVar("T")  # Type variable for the return type of the decorated function or coroutine


class EventLoopProtocol(Protocol):
    def get_loop(self) -> asyncio.AbstractEventLoop:
        ...


class LockableProtocol(Protocol):
    def get_lock(self) -> Union[threading.Lock, asyncio.Lock]:
        ...


class ThreadSafeAsyncioEventProtocol(LockableProtocol, EventLoopProtocol):
    def set_called_on_init_loop(self, value: bool) -> None:
        ...


# Decorator for checking loop conditions
def loop_conditions_decorator(callable_: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(callable_)
    def wrapper(self: EventLoopProtocol, *args, **kwargs) -> T:
        # Verify correct initialization of the class to which the decorated method belongs
        loop = self.get_loop()
        if loop is None:
            raise RuntimeError(f"There are no initial event loop in {getattr(self.__class__, 'name')}")

        if loop.is_closed() or not loop.is_running():
            raise RuntimeError("Event loop is closed or not running")

        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if current_loop is None or current_loop.is_closed() or not current_loop.is_running():
            raise RuntimeError("No event loop is running in current thread")

        kwargs["creation_loop"] = loop
        kwargs["current_loop"] = current_loop

        return callable_(self, *args, **kwargs)

    return wrapper


_LT = Callable[[Callable[..., T]], Callable[..., T]]


# Decorator for acquiring a lock before executing a function
def lock_acquisition_decorator(use_lock: bool, locking_method: str = None) -> _LT:
    """
    Decorator for acquiring a lock before executing a function.

    :param use_lock: If True, acquires a lock before executing the wrapped function.
    :param locking_method: Optional name of the method to obtain the lock object. Defaults to the `get_lock` method.
    :return: Decorated function.
    """

    def decorator(callable_: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(callable_)
        def sync_wrapper(self: LockableProtocol, *args, **kwargs) -> T:
            """
            Synchronous wrapper for the decorator.

            :param self: The instance of the class implementing the LockableProtocol.
            :param args: Positional arguments passed to the wrapped function.
            :param kwargs: Keyword arguments passed to the wrapped function.
            :return: The result of the wrapped function.
            :raises RuntimeError: If the locking method is not callable.
            :raises RuntimeError: If the locking method returns a coroutine lock.
            """
            if use_lock:
                lock: Callable[[], threading.Lock] = getattr(self, locking_method, None)
                if locking_method is None:
                    lock = self.get_lock
                if not callable(lock):
                    raise RuntimeError(f"Locking method {locking_method} is not callable")

                lock_instance = lock()
                if isinstance(lock_instance, asyncio.Lock):
                    raise RuntimeError("Locking method cannot be a coroutine function")

                with lock_instance:
                    return callable_(self, *args, **kwargs)
            else:
                return callable_(self, *args, **kwargs)

        @functools.wraps(callable_)
        async def async_wrapper(self: LockableProtocol, *args, **kwargs) -> T:
            """
            Asynchronous wrapper for the decorator.

            :param self: The instance of the class implementing the LockableProtocol.
            :param args: Positional arguments passed to the wrapped function.
            :param kwargs: Keyword arguments passed to the wrapped function.
            :return: The result of the wrapped function.
            :raises RuntimeError: If the locking method is not callable.
            """
            if use_lock:
                lock: Callable[[], Union[threading.Lock, asyncio.Lock]] = getattr(self, locking_method, None)
                if locking_method is None:
                    lock = self.get_lock
                if not callable(lock):
                    raise RuntimeError(f"Locking method {locking_method} is not callable")

                lock_instance = lock()
                if isinstance(lock_instance, asyncio.Lock):
                    async with lock_instance:
                        return await callable_(self, *args, **kwargs)

                with lock_instance:
                    return await callable_(self, *args, **kwargs)
            else:
                return await callable_(self, *args, **kwargs)

        if asyncio.iscoroutinefunction(callable_):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Decorator that ensures thread safety for synchronous functions
def ensure_thread_safety_soon(use_lock: bool = True) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A decorator that ensures thread safety for synchronous functions.
    When use_lock is True, the lock is acquired before executing the function.

    :param bool use_lock: If True, acquire the lock before executing the function.
    :return: A decorator that ensures thread safety for synchronous functions.
    """

    def decorator(callable_: Callable[..., T]) -> Callable[..., T]:
        @loop_conditions_decorator
        @lock_acquisition_decorator(use_lock)
        @functools.wraps(callable_)
        def execute_call(self: ThreadSafeAsyncioEventProtocol, *args, **kwargs) -> T:
            # Check if the current event loop is the same as the one that created the instance.
            loop = kwargs.pop("creation_loop")
            current_loop = kwargs.pop("current_loop")

            if loop is current_loop:
                self.set_called_on_init_loop(True)
                return callable_(self, *args, **kwargs)
            else:
                # For synchronous functions, use call_soon_threadsafe and create a Future to get the result.
                self.set_called_on_init_loop(False)
                future: concurrent.futures.Future[T] = concurrent.futures.Future()

                def callback():
                    try:
                        result = callable_(self, *args, **kwargs)
                        future.set_result(result)
                    except Exception as exc:
                        future.set_exception(exc)

                loop.call_soon_threadsafe(callback)
                return future.result()

        return execute_call

    return decorator


AsyncFunction = Callable[..., Awaitable[T]]


# Decorator that ensures thread safety for synchronous functions
def ensure_thread_safety_coroutine(use_lock: bool = True) -> Callable[[AsyncFunction], AsyncFunction]:
    """
    A decorator that ensures thread safety for synchronous functions.
    When use_lock is True, the lock is acquired before executing the function.

    :param bool use_lock: If True, acquire the lock before executing the function.
    :return: A decorator that ensures thread safety for synchronous functions.
    """

    def decorator(callable_: AsyncFunction) -> AsyncFunction:
        @loop_conditions_decorator
        @lock_acquisition_decorator(use_lock)
        @functools.wraps(callable_)
        async def execute_call(self: ThreadSafeAsyncioEventProtocol, *args, **kwargs) -> Union[T, None]:
            # Check if the current event loop is the same as the one that created the instance.
            loop = kwargs.pop("creation_loop")
            current_loop = kwargs.pop("current_loop")

            if loop is current_loop:
                self.set_called_on_init_loop(True)
                return await callable_(self, *args, **kwargs)
            else:
                # For asynchronous functions, use run_coroutine_threadsafe.
                self.set_called_on_init_loop(False)
                future = asyncio.run_coroutine_threadsafe(callable_(self, *args, **kwargs), loop)
                return future.result()

        return execute_call

    return decorator


class ThreadSafeAsyncioEvent(ThreadSafeAsyncioEventProtocol):
    """
    A thread-safe version of the asyncio.Event class that supports communication
    across multiple event loops running in separate threads.

    This class encapsulates the functionality of the asyncio.Event class, with
    additional thread safety and cross-event-loop communication support using
    the `asyncio.run_coroutine_threadsafe()` and `call_soon_threadsafe` methods.

    :param loop: Optional asyncio event loop. If not provided, the current event loop is used.
    :type loop: Optional[asyncio.AbstractEventLoop]

    Usage:

    .. code-block:: python

        async def main():
            ts_event = ThreadSafeAsyncioEvent()

            async def setter():
                await asyncio.sleep(1)
                ts_event.set()
                print("Event set")

            async def waiter():
                print("Waiting for event")
                ts_event.wait()
                print("Event received")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                await asyncio.gather(
                    asyncio.get_event_loop().run_in_executor(executor, run_in_thread, setter),
                    asyncio.get_event_loop().run_in_executor(executor, run_in_thread, waiter),
                )

        if __name__ == "__main__":
            asyncio.run(main())
    """

    __slots__ = (
        "_loop",
        "_current_loop",
        "_event",
        "_lock",
        "_is_called_on_init_loop"
    )

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        self._loop: asyncio.AbstractEventLoop = loop or asyncio.get_event_loop()
        self._current_loop: Optional[asyncio.AbstractEventLoop] = None
        self._event: asyncio.Event = asyncio.Event()
        self._lock: threading.Lock = threading.Lock()

        self._is_called_on_init_loop: Optional[bool] = None

    def __repr__(self) -> str:
        return f"<ThreadSafeAsyncioEvent loop={self._loop!r} is_set={self.is_set()}>"

    # All methods below are thread-safe and can be called from any thread

    def get_loop(self) -> asyncio.AbstractEventLoop:
        """
        Returns the event loop that created the instance.

        :return: The event loop that created the instance.
        """
        return self._loop

    def get_lock(self) -> threading.Lock:
        """
        Returns the lock used by the instance.

        :return: The lock used by the instance.
        """
        return self._lock

    def called_on_init_loop(self) -> bool:
        """
        Returns True if the method was called on the event loop that created the instance.

        :return: True if the method was called on the event loop that created the instance.
        """
        return self._is_called_on_init_loop

    def set_called_on_init_loop(self, value: bool) -> None:
        """
        Sets the value of the called_on_init_loop property.

        :param value: The value to set.
        :type value: bool
        """
        self._is_called_on_init_loop = value

    @ensure_thread_safety_soon(use_lock=True)
    def set(self) -> None:
        """
        Sets the event.

        This method is thread-safe and can be called from any thread.
        """
        self._event.set()

    @ensure_thread_safety_soon(use_lock=True)
    def clear(self) -> None:
        """
        Clears the event.

        This method is thread-safe and can be called from any thread.
        """
        self._event.clear()

    @ensure_thread_safety_soon(use_lock=False)
    def is_set(self) -> bool:
        """
        Returns True if the event is set, False otherwise.

        This method is thread-safe and can be called from any thread.

        :return: True if the event is set, False otherwise.
        """
        return self._event.is_set()

    @ensure_thread_safety_coroutine(use_lock=False)
    async def wait(self) -> None:
        """
        Blocks until the event is set.

        This method is thread-safe and can be called from any thread.
        """
        await self._event.wait()


# Example usage:
async def main():
    ts_event = ThreadSafeAsyncioEvent()

    def run_in_thread(func: Callable[..., Any], *args, **kwargs) -> Any:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(func(*args, **kwargs))

    async def setter():
        await asyncio.sleep(1)
        ts_event.set()
        print("Event set")

    async def waiter():
        print("Waiting for event")
        ts_event.wait()
        print("Event received")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        await asyncio.gather(
            asyncio.get_event_loop().run_in_executor(executor, run_in_thread, setter),
            asyncio.get_event_loop().run_in_executor(executor, run_in_thread, waiter),
        )


if __name__ == "__main__":
    asyncio.run(main())
