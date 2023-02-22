import asyncio
import gc
import sys
import threading
from typing import Any, AsyncGenerator, Coroutine, Dict, Optional, Union

import aiohttp
from aiohttp import ClientSession

from hummingbot.core.utils.weak_singleton_metaclass import ClassNotYetInstantiatedError, WeakSingletonMetaclass


def _increment_ref_count(*, thread_id: int, ref_count: Dict[int, int]):
    """Increments the reference count for the given thread."""
    if thread_id not in ref_count:
        ref_count[thread_id] = 0
    ref_count[thread_id] = ref_count[thread_id] + 1


def _decrement_ref_count(*, thread_id: int, ref_count: Dict[int, int]):
    """decrements the reference count for the given thread."""
    if thread_id not in ref_count:
        raise RuntimeError(f"Thread {thread_id} is not in ref_count. Something went badly.")
    else:
        ref_count[thread_id] = ref_count[thread_id] - 1


class NotWithinAsyncFrameworkError(Exception):
    pass


class PersistentClientSessionCounter(metaclass=WeakSingletonMetaclass):
    """ A persistent class to manage aiohttp client sessions.

    The class uses a shared client object to manage aiohttp client sessions,
    with a reference count to keeps track of the number of active sessions.
    The shared client is automatically cleaned up when the reference count reaches 0.
    """
    _session_mutex: asyncio.Lock = asyncio.Lock()
    _shared_client_session: Dict[int, Union[aiohttp.ClientSession, None]] = dict()
    _init_ref_count: Dict[int, int] = dict()
    _context_ref_count: Dict[int, int] = dict()
    _all_ref_count: Dict[int, int] = dict()

    __slots__ = ()

    @classmethod
    def __del__(cls):
        print("del")

    @classmethod
    def __init__(cls, **kwargs):
        """Noncontextual instantiation of the aiohttp.ClientSession instance.

        :param kwargs: Keyword arguments for the aiohttp.ClientSession constructor.
        :return: The aiohttp.ClientSession instance associated with the current thread.
        :rtype: Coroutine[Any, Any, aiohttp.ClientSession]
        """
        print("init")
        assert cls._class_is_created_or_raise()

        # Get the current running loop
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            raise NotWithinAsyncFrameworkError(
                "The event loop is not running. Do not instantiate a PersistentClientSession "
                "outside of an async context.")

        thread_id: int = threading.get_ident()

        # If the session is not created, create it
        cls._create_session(thread_id=thread_id, **kwargs)

        _increment_ref_count(thread_id=thread_id, ref_count=cls._init_ref_count)

        # Register the references to the session
        cls._all_ref_count[thread_id] = sys.getrefcount(cls._shared_client_session[thread_id]) - 1

    @classmethod
    async def __aenter__(cls, **kwargs) -> Optional[ClientSession]:
        """
        Context manager entry method.

        :param kwargs: Keyword arguments for the aiohttp.ClientSession constructor.
        :return: The aiohttp.ClientSession instance associated with the current thread.
        :rtype: Coroutine[Any, Any, aiohttp.ClientSession]
        """
        assert cls._class_is_created_or_raise()

        thread_id: int = threading.get_ident()
        # This method is either called immediately after __init__() or with a reference
        # Increment the context reference count
        _increment_ref_count(thread_id=thread_id, ref_count=cls._context_ref_count)

        # If the session is not created, create it
        await cls._async_create_session(thread_id=thread_id, **kwargs)

        # Register the references to the session
        cls._all_ref_count[thread_id] = sys.getrefcount(cls._shared_client_session[thread_id]) - 1

        await asyncio.sleep(0)
        return cls._shared_client_session[thread_id]

    @classmethod
    async def _async_create_session(cls, *, thread_id: int, **kwargs):
        """
        Create a new session for the given thread id.

        :param int thread_id: The identifier for the thread to create a session for.
        :param kwargs: Keyword arguments for the aiohttp.ClientSession constructor.
        """
        assert cls._class_is_created_or_raise()

        # If the session is already created, we can return
        if cls._thread_has_session(thread_id=thread_id):
            if not cls._shared_client_session[thread_id].closed:
                await asyncio.sleep(0)
                return
            else:
                cls._cleanup_session(thread_id=thread_id)

        # Otherwise, we need to create the session
        async with cls._session_mutex:
            # Another thread could have created the session
            # before we acquired the lock. So check that the
            # session is still nonexistent.
            cls._create_session(thread_id=thread_id, **kwargs)
            await asyncio.sleep(0)

        assert cls._shared_client_session[thread_id] is not None
        # assert not cls._shared_client_session[thread_id].closed

    @classmethod
    def _create_session(cls, *, thread_id: int, **kwargs):
        """
        Create a new session for the given thread id.

        :param int thread_id: The identifier for the thread to create a session for.
        :param kwargs: Keyword arguments for the aiohttp.ClientSession constructor.
        """
        assert cls._class_is_created_or_raise()

        # If the session is already created, we can return
        if cls._thread_has_session(thread_id=thread_id):
            if not cls._shared_client_session[thread_id].closed:
                return
            else:
                cls._cleanup_session(thread_id=thread_id)

        try:
            cls._shared_client_session[thread_id] = aiohttp.ClientSession(**kwargs)
        except RuntimeError as e:
            cls._cleanup_session(thread_id=thread_id)
            print(f"Error creating aiohttp session: {e}")
            raise e

        assert cls._shared_client_session[thread_id] is not None
        # assert not cls._shared_client_session[thread_id].closed

    @classmethod
    async def __aexit__(cls, exc_type, exc_val, exc_tb):
        """
        Context manager exit method.

        Decrements the reference count for the current thread, and cleans up the
        shared client if the reference count reaches 0.

        :param exc_type: Type of exception raised.
        :param exc_val: Value of exception raised.
        :param exc_tb: Traceback of exception raised.
        """
        assert cls._class_is_created_or_raise()

        thread_id: int = threading.get_ident()
        # Decrement the ref count for the current thread. __init__ is called when entering the context
        # manager with the Class call
        _decrement_ref_count(thread_id=thread_id, ref_count=cls._context_ref_count)
        _decrement_ref_count(thread_id=thread_id, ref_count=cls._init_ref_count)

        # Register out of context references (Likely a minimum of 1)
        reference_offset = sys.getrefcount(cls._shared_client_session[thread_id]) - 1 - cls._all_ref_count[thread_id]

        # If the ref count is 0 and there are no extra references, we can safely close the session
        if cls._context_ref_count[thread_id] == 0 and reference_offset > 0:
            # ref_count is zero, but an out-of-context reference exists,
            # so we can't clean up yet, let's leave that task to the finalizer
            print(f"Deferring cleanup for thread {thread_id} to the class collector."
                  f"The async context manager was either returned or the PersistentClientSession"
                  f" instantiated outside its context.")
        else:
            # If the ref count is 0, we can safely close the session
            await cls._shared_client_session[thread_id].close()
            await cls._async_cleanup_session(thread_id=thread_id)
        await asyncio.sleep(0)

    @classmethod
    def _class_is_created_or_raise(cls):
        """
        Raises an error if the class has not been instantiated yet.

        :raises ClassNotYetInstantiatedError: If the class has not been instantiated yet.
        :return: True if the class has been instantiated.
        :rtype: bool
        """
        if not cls.is_class_instantiated(cls):
            raise ClassNotYetInstantiatedError("Class not created yet."
                                               f"The methods of `{cls.__class__.__name__}'"
                                               " can only be called after this 'Singleton'"
                                               " class has been instantiated at least once.")
        return True

    @classmethod
    def _current_thread_has_session(cls):
        """Checks if the current thread has a live session"""
        assert cls._class_is_created_or_raise()

        thread_id: int = threading.get_ident()
        return cls._thread_has_session(thread_id=thread_id)

    @classmethod
    def _thread_has_session(cls, *, thread_id: int):
        """Checks if the thread has a live session"""
        assert cls._class_is_created_or_raise()

        return thread_id in cls._shared_client_session and cls._shared_client_session[thread_id] is not None

    @classmethod
    def _is_live_session(cls, *, thread_id: int):
        """Checks if the thread has a live session"""
        assert cls._class_is_created_or_raise()

        return cls._thread_has_session(thread_id=thread_id) and not cls._shared_client_session[thread_id].closed

    @classmethod
    def _is_closed_session(cls, *, thread_id: int):
        """Checks if the has a closed session."""
        assert cls._class_is_created_or_raise()

        return cls._thread_has_session(thread_id=thread_id) and cls._shared_client_session[thread_id].closed

    @classmethod
    def _increment_context_ref_count(cls, *, thread_id: int):
        """Increments the reference count for the given thread."""
        assert cls._class_is_created_or_raise()

        if thread_id not in cls._context_ref_count:
            cls._context_ref_count[thread_id] = 0
        cls._context_ref_count[thread_id] = cls._context_ref_count[thread_id] + 1

    @classmethod
    def _decrement_context_ref_count(cls, *, thread_id: int):
        """Decrements the reference count for the given thread."""
        assert cls._class_is_created_or_raise()

        if thread_id not in cls._context_ref_count:
            raise RuntimeError(f"Thread {thread_id} is not in ref_count. Something went badly.")
        else:
            cls._context_ref_count[thread_id] = cls._context_ref_count[thread_id] - 1

    @classmethod
    async def _async_clear_to_non_instantiated(cls):
        """ Clears the class to a non-instantiated state. This method is called by the Singleton metaclass
        when the class is deleted (basically, when the singleton instance is deleted).
        """
        if cls.is_class_instantiated():
            for thread_id in cls._shared_client_session:
                await cls._async_cleanup_session(thread_id=thread_id)
        # Recommended wait for closing the ClientSession
        await asyncio.sleep(0)

    @classmethod
    def _cleanup_session(cls, *, thread_id: int):
        """Cleans up the class when the session was closed"""
        assert cls._class_is_created_or_raise()
        assert cls._thread_has_session(thread_id=thread_id)

        cls._shared_client_session[thread_id] = None
        cls._context_ref_count[thread_id] = 0

    @classmethod
    async def _async_cleanup_session(cls, *, thread_id: int):
        """Cleans up the shared client for the given thread, likely when exiting the context manager."""
        assert cls._class_is_created_or_raise()
        assert cls._thread_has_session(thread_id=thread_id)

        if cls._is_live_session(thread_id=thread_id):
            # Close the aiohttp Session
            await cls._shared_client_session[thread_id].close()
        cls._cleanup_session(thread_id=thread_id)
        await asyncio.sleep(0)

    @classmethod
    async def async_yield_session_in_context(cls) -> AsyncGenerator[None, aiohttp.ClientSession]:
        """
        Request out-of-context access to the shared client.

        :param int thread_id: The identifier for the thread to clean up the shared client for.
        """
        assert cls._class_is_created_or_raise()

        async with PersistentClientSession() as session:
            yield session

    @classmethod
    async def async_get_session(cls) -> Coroutine[Any, Any, aiohttp.ClientSession]:
        """
        Request out-of-context access to the shared client.
        """
        assert cls._class_is_created_or_raise()

        # thread_id = threading.get_ident()
        # await cls._create_session(thread_id=thread_id)
        # cls._increment_ref_count(thread_id=thread_id)
        # print(f"     count {thread_id}: {weakref.getweakrefcount(cls._shared_session[thread_id])}")
        session_iter: AsyncGenerator[None, aiohttp.ClientSession] = cls.async_yield_session_in_context()
        session: Coroutine[aiohttp.ClientSession]
        async for session in session_iter:
            return session

    @classmethod
    def __is_finalized(cls, *, thread_id: int):
        """
        Check if the session for the given thread has been finalized.

        :param int thread_id: The identifier for the thread to check.
        :return: True if the session has been finalized, False otherwise.
        :rtype: bool
        """
        assert cls._class_is_created_or_raise()

        return thread_id in cls._finalizer and not cls._finalizer[thread_id].alive

    @classmethod
    def __finalize(cls, *, thread_id: int):
        """
        Check if the session for the given thread has been finalized.

        :param int thread_id: The identifier for the thread to check.
        :return: True if the session has been finalized, False otherwise.
        :rtype: bool
        """
        print(f"Finalizing session for class {cls.__name__} {vars(cls)}")
        # assert cls._class_is_created_or_raise()
        print(f"Finalizing session for thread {thread_id}")
        cls._cleanup(thread_id=thread_id)
        cls._finalizer[thread_id] = None

    # @classmethod
    # def get_client_in_context(cls) -> aiohttp.ClientSession:
    #    """
    #    Request out-of-context access to the shared client. This method enter the context manager, but 'returns'


class PersistentClientSession(metaclass=WeakSingletonMetaclass):
    """ A persistent class to manage aiohttp client sessions.

    The class uses a shared client object to manage aiohttp client sessions,
    with a reference count to keeps track of the number of active sessions.
    The shared client is automatically cleaned up when the reference count reaches 0.
    """
    _session_mutex: Dict[int, asyncio.Lock] = dict()
    _shared_client_session: Dict[int, Union[aiohttp.ClientSession, None]] = dict()
    _kwargs_client_session: Dict[int, Dict[str, Any]] = dict()

    _init_count: Dict[int, int] = dict()
    _ref_count: Dict[int, int] = dict()
    _in_context_count: Dict[int, int] = dict()

    def __del__(self):
        print(f"Deleting class {self.__class__.__name__}")
        print(f"   refs: {gc.get_referrers(self)}")
        thread_id = threading.get_ident()
        if self.has_live_session(thread_id=thread_id):
            # The session was not closed cleanly by the user (or the context manager)
            # This is not a good practice, but we can try to close the session from another loop
            # if the current loop is already closed (which is the case when the class is deleted)
            self._cleanup_in_thread(thread_id=thread_id)

    def __post_call__(self, **kwargs):
        print("post call")

    def __init__(self, **kwargs):
        """Noncontextual instantiation of the aiohttp.ClientSession instance.

        :param kwargs: Keyword arguments for the aiohttp.ClientSession constructor.
        :return: The aiohttp.ClientSession instance associated with the current thread.
        :rtype: Coroutine[Any, Any, aiohttp.ClientSession]
        """
        print("init")
        thread_id: int = threading.get_ident()
        self._kwargs_client_session[thread_id] = kwargs
        self._session_mutex[thread_id] = asyncio.Lock()

        if not self.__class__.is_class_instantiated(self.__class__):  # type: ignore
            # Get the current running loop
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                raise NotWithinAsyncFrameworkError(
                    "The event loop is not running. Do not instantiate a PersistentClientSession "
                    "outside of an async context.")

            # Create the session if it does not exist
            self.__create_session(thread_id=thread_id, **kwargs)
            self._in_context_count[thread_id] = 0
            self._ref_count[thread_id] = 0
            self._init_count[thread_id] = 0

        self._init_count[thread_id] = self._init_count.get(thread_id, 0) + 1
        self._ref_count[thread_id] = sys.getrefcount(self._shared_client_session[thread_id]) - 1

        print(
            f"   counts: {self._ref_count[thread_id]}:{self._init_count[thread_id]}:{self._in_context_count[thread_id]}")

    async def session(self) -> Optional[ClientSession]:
        thread_id: int = threading.get_ident()
        # Create the session if it does not exist
        await self.__async_create_session(thread_id=thread_id)
        return self._shared_client_session[thread_id]

    async def close(self):
        assert self._class_is_created_or_raise()
        await self.__async_cleanup_session(thread_id=threading.get_ident())

    async def __aenter__(self, **kwargs) -> Optional[ClientSession]:
        """
        Context manager entry method. There are a few cases to consider for __aexit__:
            1. We enter context with a reference to the singleton instance
                The instance already exists, it should have a live session, the init_count >= 1
                If there is a 'as' clause, the reference count increases after __aenter__():
                    We can skip closing the session
                If there is no 'as' clause, the reference count remains the same:
                    How to know that we should not close the session?
            2. We enter context with a call without 'as'
                The instance already exist, and it should have a live session
            3. We enter context with a call and assign it with 'as'

        :param kwargs: Keyword arguments for the aiohttp.ClientSession constructor.
        :return: The aiohttp.ClientSession instance associated with the current thread.
        :rtype: Coroutine[Any, Any, aiohttp.ClientSession]
        """
        print("__aenter__")
        assert self._class_is_created_or_raise()

        thread_id: int = threading.get_ident()

        # If the session is not created, create it
        await self.__async_create_session(thread_id=thread_id, **kwargs)

        self._ref_count[thread_id] = sys.getrefcount(self._shared_client_session[thread_id]) - 1
        self._in_context_count[thread_id] = self._in_context_count.get(thread_id, 0) + 1

        print(
            f"   counts: {self._ref_count[thread_id]}:{self._init_count[thread_id]}:{self._in_context_count[thread_id]}")
        await asyncio.sleep(0)
        return self._shared_client_session[thread_id]

    async def __async_create_session(self, *, thread_id: int):
        """
        Create a new session for the given thread id.

        :param int thread_id: The identifier for the thread to create a session for.
        """
        # If the session is already created, we can return
        if self.__thread_has_session(thread_id=thread_id):
            if not self._shared_client_session[thread_id].closed:
                await asyncio.sleep(0)
                return
            else:
                await self._cleanup_closed_session(thread_id=thread_id)

        # Otherwise, we need to create the session
        async with self._session_mutex[thread_id]:
            self.__create_session(thread_id=thread_id)
            await asyncio.sleep(0)

        assert self._shared_client_session[thread_id] is not None
        # assert not cls._shared_client_session[thread_id].closed

    def __create_session(self, *, thread_id: int):
        """
        Create a new session for the given thread id.

        :param int thread_id: The identifier for the thread to create a session for.
        """
        # If the session is already created (by another async call) and not closed, we can return
        if self.__thread_has_session(thread_id=thread_id):
            if not self._shared_client_session[thread_id].closed:
                return
            # However, if the session is closed, we need to clean it up then recreate it
            else:
                self._cleanup_in_thread(thread_id=thread_id)
        try:
            self._shared_client_session[thread_id] = aiohttp.ClientSession(**self._kwargs_client_session[thread_id])
        except RuntimeError as e:
            self._cleanup_in_thread(thread_id=thread_id)
            print(f"Error creating aiohttp session: {e}")
            raise e

        assert self._shared_client_session[thread_id] is not None
        # assert not cls._shared_client_session[thread_id].closed

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit method. There are a few cases to consider for __aexit__:
            1. We enter context with a reference to the singleton instance
                If there is a 'as' clause, the reference count increases after __aenter__():
                    We can skip closing the session on references_offset >= 1
                If there is no 'as' clause, the reference count remains the same:
                    How to know that we should not close the session?
            2. We enter context with a call without 'as'
                The instance already exist, and it should have a live session
            3. We enter context with a call and assign it with 'as'

        Decrements the reference count for the current thread, and cleans up the
        shared client if the reference count reaches 0.

        :param exc_type: Type of exception raised.
        :param exc_val: Value of exception raised.
        :param exc_tb: Traceback of exception raised.
        """
        assert self._class_is_created_or_raise()

        print("__aexit__")
        thread_id: int = threading.get_ident()

        references_offset = sys.getrefcount(self._shared_client_session[thread_id]) - 1 - self._ref_count[thread_id]
        # Register out of context references (Likely a minimum of 1)
        self._in_context_count[thread_id] = self._in_context_count.get(thread_id) - 1

        # Are there any other references?
        print(
            f"   counts: {self._ref_count[thread_id]}:{self._init_count[thread_id]}:{self._in_context_count[thread_id]}")
        print(f"  ref offset: {references_offset}")

        # If the ref count is 0 and there are no extra references, we can safely close the session
        if self._in_context_count[thread_id] >= 1 or references_offset >= 1:
            # ref_count is zero, but an out-of-context reference exists,
            # so we can't clean up yet, let's leave that task to the finalizer
            print(f"Deferring cleanup for thread {thread_id} to the class collector."
                  f"The async context manager was either returned or the PersistentClientSession"
                  f" instantiated outside its context.")
        else:
            # If the ref count is 0, we can safely close the session
            await self._shared_client_session[thread_id].close()
            await self.__async_cleanup_session(thread_id=thread_id)
        await asyncio.sleep(0)

    def _class_is_created_or_raise(self):
        """
        Raises an error if the class has not been instantiated yet.

        :raises ClassNotYetInstantiatedError: If the class has not been instantiated yet.
        :return: True if the class has been instantiated.
        :rtype: bool
        """
        if not self.__class__.is_class_instantiated(self.__class__):
            raise ClassNotYetInstantiatedError("Class not created yet."
                                               f"The methods of `{self.__class__.__name__}'"
                                               " can only be called after this 'Singleton'"
                                               " class has been instantiated at least once.")
        return True

    def __thread_has_session(self, *, thread_id: int):
        """Checks if the thread has a live session"""
        return thread_id in self._shared_client_session and self._shared_client_session[thread_id] is not None

    def has_live_session(self, *, thread_id: int):
        """Checks if the thread has a live session"""
        assert self._class_is_created_or_raise()
        return self.__thread_has_session(thread_id=thread_id) and not self._shared_client_session[thread_id].closed

    @classmethod
    async def _async_clear_to_non_instantiated(cls):
        """ Clears the class to a non-instantiated state. This method is called by the Singleton metaclass
        when the class is deleted (basically, when the singleton instance is deleted).
        """
        if cls.is_class_instantiated(cls):
            for thread_id in cls._shared_client_session:
                await cls.__async_cleanup_session(thread_id=thread_id)
        # Recommended wait for closing the ClientSession
        await asyncio.sleep(0)

    async def __async_cleanup_session(self, *, thread_id: int):
        """Cleans up the class when the session was closed"""
        assert self._class_is_created_or_raise()

        print(f"Closing session for thread {thread_id}")
        if self.has_live_session(thread_id=thread_id):
            await self._shared_client_session[thread_id].close()
            print("Closed async")
        self._shared_client_session[thread_id] = None
        return

    async def _cleanup_closed_session(self, *, thread_id: int):
        """Cleans up the class when the session was closed"""
        assert self._class_is_created_or_raise()
        self._shared_client_session[thread_id] = None

    def _cleanup_in_thread(self, *, thread_id: int):
        """Runs the cleanup using a call from a different thread on an event loop created locally
        to the main thread."""
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=False).start()
        asyncio.run_coroutine_threadsafe(self.__async_cleanup_session(thread_id=thread_id), loop=loop).result()
        loop.call_soon_threadsafe(loop.stop)

    @classmethod
    async def async_get_session(cls) -> Coroutine[Any, Any, aiohttp.ClientSession]:
        """
        Request out-of-context access to the shared client.
        """
        assert cls._class_is_created_or_raise()

        # thread_id = threading.get_ident()
        # await cls._create_session(thread_id=thread_id)
        # cls._increment_ref_count(thread_id=thread_id)
        # print(f"     count {thread_id}: {weakref.getweakrefcount(cls._shared_session[thread_id])}")
        session_iter: AsyncGenerator[None, aiohttp.ClientSession] = cls.async_yield_session_in_context()
        session: Coroutine[aiohttp.ClientSession]
        async for session in session_iter:
            return session

    @classmethod
    def __is_finalized(cls, *, thread_id: int):
        """
        Check if the session for the given thread has been finalized.

        :param int thread_id: The identifier for the thread to check.
        :return: True if the session has been finalized, False otherwise.
        :rtype: bool
        """
        assert cls._class_is_created_or_raise()

        return thread_id in cls._finalizer and not cls._finalizer[thread_id].alive

    @classmethod
    def __finalize(cls, *, thread_id: int):
        """
        Check if the session for the given thread has been finalized.

        :param int thread_id: The identifier for the thread to check.
        :return: True if the session has been finalized, False otherwise.
        :rtype: bool
        """
        print(f"Finalizing session for class {cls.__name__} {vars(cls)}")
        # assert cls._class_is_created_or_raise()
        print(f"Finalizing session for thread {thread_id}")
        cls._cleanup(thread_id=thread_id)
        cls._finalizer[thread_id] = None

    # @classmethod
    # def get_client_in_context(cls) -> aiohttp.ClientSession:
    #    """
    #    Request out-of-context access to the shared client. This method enter the context manager, but 'returns'

#
#    :return: The shared client session for the current thread.
#    """
#    assert cls._class_is_created_or_raise()
#
#    if cls._is_live_session(thread_id=threading.get_ident()):
#        return cls.async_get_client_in_context()
#    async with PersistentClientSession() as client:
#        return await client
#
# Registering a finalizer for all sessions, in and out of context
# cls._finalizer[thread_id] = weakref.finalize(cls._shared_session[thread_id],
#                                             cls.__finalize, thread_id=thread_id)
# print(f"Registering finalizer for {thread_id}: {weakref.getweakrefcount(cls._shared_session[thread_id])}")
