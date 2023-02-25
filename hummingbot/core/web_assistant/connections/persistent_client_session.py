import asyncio
import gc
import sys
import threading
from functools import partial
from typing import Any, Callable, Coroutine, Dict, Union

from aiohttp import ClientSession

from hummingbot.core.utils.weak_singleton_metaclass import ClassNotYetInstantiatedError, WeakSingletonMetaclass


class NotWithinAsyncFrameworkError(Exception):
    pass


class PersistentClientSession(metaclass=WeakSingletonMetaclass):
    """ A persistent class to manage aiohttp client sessions.

    The class uses a shared client object to manage aiohttp client sessions,
    with a reference count to keeps track of the number of active sessions.
    The shared client is automatically cleaned up when the reference count reaches 0.
    """
    _sessions_mutex: Dict[int, asyncio.Lock] = dict()
    _shared_client_sessions: Dict[int, Union[ClientSession, None]] = dict()
    _kwargs_client_sessions: Dict[int, Dict[str, Any]] = dict()
    _original_sessions_close: Dict[int, Union[Callable[[], Coroutine[Any, Any, None]], None]] = dict()

    _init_count: Dict[int, int] = dict()
    _ref_count: Dict[int, int] = dict()
    _in_context_count: Dict[int, int] = dict()

    def __del__(self):
        print(f"Deleting class {self.__class__.__name__}")
        print(f"   refs: {gc.get_referrers(self)}")
        thread_id = threading.get_ident()
        if self.__class__.is_instantiated():
            if self.has_live_session(thread_id=thread_id):
                # The session was not closed cleanly by the user (or the context manager)
                # This is not a good practice, but we can try to close the session from another loop
                # if the current loop is already closed (which is the case when the class is deleted)
                self.__cleanup_in_thread(thread_id=thread_id)

    def __init__(self, **kwargs):
        print("__init__", kwargs)
        thread_id: int = threading.get_ident()
        self._sessions_mutex[thread_id] = asyncio.Lock()
        self._shared_client_sessions[thread_id] = None
        self._original_sessions_close[thread_id] = None
        self._kwargs_client_sessions[thread_id] = kwargs

        self._ref_count[thread_id] = 0
        self._in_context_count[thread_id] = 0

    def __call__(self, **kwargs) -> ClientSession:
        """Noncontextual instantiation of the ClientSession instance.

        :param kwargs: Keyword arguments for the ClientSession constructor.
        :return: The ClientSession instance associated with the current thread.
        :rtype: Coroutine[Any, Any, ClientSession]
        """
        print("__call__")
        assert self.is_instantiated_or_raise()

        self.__in_running_loop_or_raise()

        thread_id: int = threading.get_ident()
        self._kwargs_client_sessions[thread_id] = kwargs

        # Create a session if one is not in the process of being created on the event loop
        self._get_or_create_session(thread_id=thread_id, should_be_locked=False)

        self._ref_count[thread_id] = sys.getrefcount(self._shared_client_sessions[thread_id]) - 1
        return self._shared_client_sessions[thread_id]

    async def create_session(self, *, thread_id: int):
        """
        Create a new session for the given thread id.

        :param int thread_id: The identifier for the thread to create a session for.
        """
        # If the session is already created (by another async call) and not closed, we can return
        if self.has_live_session(thread_id=thread_id):
            await asyncio.sleep(0)
            return

        self._cleanup_closed_session(thread_id=thread_id)

        # Otherwise, we need to create the session
        async with self._sessions_mutex[thread_id]:
            self._get_or_create_session(thread_id=thread_id, should_be_locked=True)
            await asyncio.sleep(0)

        assert self._shared_client_sessions[thread_id] is not None
        assert not self._shared_client_sessions[thread_id].closed

    def _get_or_create_session(self, *, thread_id: int, should_be_locked: bool = True):
        """
        Create a new session if needed for the given thread id.

        :param int thread_id: The identifier for the thread to create a session for.
        :param bool should_be_locked: Whether the session mutex should be locked.
        :raises RuntimeError: Collision between sync and async calls.
        """
        if self.has_live_session(thread_id=thread_id):
            return

        # Bitwise AND to check if the lock is locked when it should be
        # For instance, in the case of a sync call, the lock should not be locked:
        # meaning, no async call in the process of creating a session
        if should_be_locked == self._sessions_mutex[thread_id].locked():
            try:
                self._shared_client_sessions[thread_id] = ClientSession(**self._kwargs_client_sessions[thread_id])
                self._original_sessions_close[thread_id] = self._shared_client_sessions[thread_id].close
                self._shared_client_sessions[thread_id].close = partial(self._triggered_close, thread_id=thread_id)

                self._ref_count[thread_id] = sys.getrefcount(self._shared_client_sessions[thread_id]) - 1
            except RuntimeError as e:
                # The session failed to be created
                self._cleanup_closed_session(thread_id=thread_id)
                raise e
        elif not should_be_locked:
            raise RuntimeError("The session is already being created in async context."
                               "This is not allowed in sync context and a design flaw")

    async def open(self, **kwargs) -> ClientSession:
        """Request out-of-context access to the shared client."""
        assert self.is_instantiated_or_raise()
        thread_id: int = threading.get_ident()
        # Update the kwargs if they are different
        if kwargs and kwargs != self._kwargs_client_sessions[thread_id]:
            self._kwargs_client_sessions[thread_id] = kwargs
        await self.create_session(thread_id=thread_id)
        return self._shared_client_sessions[thread_id]

    async def close(self):
        """Close the shared client session."""
        assert self.is_instantiated_or_raise()
        await self.async_session_cleanup(thread_id=threading.get_ident())

    async def _triggered_close(self, *, thread_id: int):
        print("Closing session from ClientSession triggers cleanup in PersistentClientSession")
        await self._original_sessions_close[thread_id]()
        self._cleanup_closed_session(thread_id=thread_id)

    def has_live_session(self, *, thread_id: int) -> bool:
        """Checks if the thread has a live session"""
        assert self.is_instantiated_or_raise()
        return self.thread_has_session(thread_id=thread_id) and not self._shared_client_sessions[thread_id].closed

    async def __aenter__(self) -> ClientSession:
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

        :param kwargs: Keyword arguments for the ClientSession constructor.
        :return: The ClientSession instance associated with the current thread.
        :rtype: Coroutine[Any, Any, ClientSession]
        """
        print("__aenter__")
        assert self.is_instantiated_or_raise()
        thread_id = threading.get_ident()
        self._in_context_count[thread_id] = sys.getrefcount(self._shared_client_sessions[thread_id]) - 1
        return await self.open()

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
        assert self.is_instantiated_or_raise()

        print("__aexit__")
        thread_id: int = threading.get_ident()

        references_offset = sys.getrefcount(self._shared_client_sessions[thread_id]) - 1 - self._in_context_count[
            thread_id]

        # Are there any other references?
        print(
            f"   counts: {self._ref_count[thread_id]}:{self._init_count[thread_id]}:{self._in_context_count[thread_id]}")
        print(f"  ref offset: {references_offset}")

        # If the ref count is 0 and there are no extra references, we can safely close the session
        if references_offset >= 1:
            # ref_count is zero, but an out-of-context reference exists,
            # so we can't clean up yet, let's leave that task to the finalizer
            print(f"Deferring cleanup for thread {thread_id} to the class collector."
                  f"The async context manager was either returned or the PersistentClientSession"
                  f" instantiated outside its context.")
        else:
            # If the ref count is 0, we can safely close the session
            await self._shared_client_sessions[thread_id].close()
            await self.async_session_cleanup(thread_id=thread_id)
        await asyncio.sleep(0)

    def __getattr__(self, attr):
        """
        Wraps all other method calls to the underlying `aiohttp.ClientSession` instance.

        :param attr: The name of the method to call.

        Example:
        --------
        persistent_session = PersistentClientSession(session)
        response = await persistent_session.get('https://www.example.com')
        """
        thread_id: int = threading.get_ident()
        if hasattr(self._shared_client_sessions[thread_id], attr):
            return getattr(self._shared_client_sessions[thread_id], attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}',"
                                 f" nor '{self._shared_client_sessions[thread_id].__class__.__name__}'"
                                 f" object has attribute '{attr}'")

    @staticmethod
    def __in_running_loop_or_raise():
        """
        Raises an error if the current thread is not running an asyncio event loop.

        :raises NoRunningLoopError: If the current thread is not running an asyncio event loop.
        :return: True if the current thread is running an asyncio event loop.
        :rtype: bool
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            raise NotWithinAsyncFrameworkError("The event loop is not running."
                                               "This method requires a running event loop")

    def is_instantiated_or_raise(self) -> bool:
        """
        Raises an error if the class has not been instantiated yet.

        :raises ClassNotYetInstantiatedError: If the class has not been instantiated yet.
        :return: True if the class has been instantiated.
        :rtype: bool
        """
        if not self.__class__.is_instantiated():
            raise ClassNotYetInstantiatedError("Class not created yet."
                                               f"The methods of `{self.__class__.__name__}'"
                                               " can only be called after this 'Singleton'"
                                               " class has been instantiated at least once.")
        return True

    def thread_has_session(self, *, thread_id: int) -> bool:
        """
        Returns True if the thread has a session. The session may be closed.
        :param thread_id:
        :return: True if the thread has a session.
        :rtype: bool
        """
        return thread_id in self._shared_client_sessions and self._shared_client_sessions[thread_id] is not None

    async def _async_clear_to_non_instantiated(self):
        """
        This method is called by the metaclass when the class is being destroyed. It should NOT be called directly.
        It should be very rare when this method is called, as the class should only be destroyed when the reference
        count reaches 0.
        :return: None
        """
        if self.__class__.is_instantiated():
            for thread_id in self._shared_client_sessions:
                await self.async_session_cleanup(thread_id=thread_id)
        # Recommended wait for closing the ClientSession
        await asyncio.sleep(0)

    async def async_session_cleanup(self, *, thread_id: int):
        """
        Closes the ClientSession for the thread, and cleans up the thread resources.
        :param thread_id:
        :return: None
        """
        assert self.is_instantiated_or_raise()

        print(f"Closing session for thread {thread_id}")
        if self.has_live_session(thread_id=thread_id):
            await self._shared_client_sessions[thread_id].close()
            print("Closed async")
        self._cleanup_closed_session(thread_id=thread_id)

    def _cleanup_closed_session(self, *, thread_id: int):
        """
        Cleans up the thread resources if the session is closed, does nothing otherwise.
        :param thread_id:
        :return: None
        """
        assert self.is_instantiated_or_raise()
        # If the session is closed, we can safely clear it, otherwise silently ignore
        if self._shared_client_sessions[thread_id] is not None and self._shared_client_sessions[thread_id].closed:
            self._shared_client_sessions[thread_id] = None
            self._kwargs_client_sessions[thread_id] = {}

    def __cleanup_in_thread(self, *, thread_id: int):
        """
        Closes the ClientSession for the thread, and cleans up the thread resources. This uses a new event loop
        in a different thread to run the async cleanup.
        :param thread_id:
        :return: None
        """
        assert self.is_instantiated_or_raise()
        # Do we still have an event loop in the main thread?
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Is it even possible to create a new loop and close here?
            # If the main loop s closed it is already too late
            raise RuntimeError("No running event loop in the main thread. "
                               "Too late to run the cleanup in the main thread.")

        # Create a new event loop in a different thread and run the cleanup
        # A call to asyncio.run() is not possible here because the loop is already/still running
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=False).start()
        asyncio.run_coroutine_threadsafe(self.async_session_cleanup(thread_id=thread_id), loop=loop).result()
        loop.call_soon_threadsafe(loop.stop)

        # Apparently the loop cannot be closed, not sure why
        # loop.close()  # This fails to execute in tests
