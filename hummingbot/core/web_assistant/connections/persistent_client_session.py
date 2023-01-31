import asyncio
import threading
from typing import Dict, Optional, Union

import aiohttp


class ClassNotCreatedError(Exception):
    pass


class PersistentClientSession:
    """ A persistent aiohttp.ClientSession that can be shared across threads and asyncio tasks."""
    _instance: Optional["PersistentClientSession"] = None
    _lock = threading.Lock()

    _session_mutex = asyncio.Lock()
    _shared_client: Dict[int, Union[aiohttp.ClientSession, None]] = dict()
    _ref_count: Dict[int, int] = dict()
    _aenter: Dict[int, Union[aiohttp.ClientSession, None]] = dict()

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                # Another thread could have created the instance
                # before we acquired the lock. So check that the
                # instance is still nonexistent.
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    async def __aenter__(self, *args, **kwargs):
        thread_id = threading.get_ident()
        # Increment the ref count for the current thread
        self._increment_ref_count(thread_id=thread_id)
        await self._create_session(thread_id=thread_id)
        return self._aenter[thread_id]

    async def __aexit__(self, *args, **kwargs):
        thread_id = threading.get_ident()
        # Decrement the ref count for the current thread
        self._ref_count[thread_id] = self._ref_count[thread_id] - 1

        # If the ref count is 0, we clean up
        if self._ref_count[thread_id] == 0:
            # If the ref count is 0, we can safely close the session
            await self._shared_client[thread_id].__aexit__(None, None, None)
            self._cleanup(thread_id=thread_id)

    @classmethod
    def _class_is_created_or_raise(cls):
        if cls._instance is None:
            raise ClassNotCreatedError("Class not created yet. The class methods cannot be called without an instance")
        return True

    @classmethod
    async def _create_session(cls, *, thread_id: int):
        assert cls._class_is_created_or_raise()
        # If the session is already created, we can return
        if thread_id in cls._shared_client and cls._shared_client[thread_id] is not None:
            await asyncio.sleep(0)
            return

        # Otherwise, we need to create the session
        async with cls._session_mutex:
            cls._shared_client[thread_id] = aiohttp.ClientSession()
            cls._aenter[thread_id] = await cls._shared_client[thread_id].__aenter__()
        assert cls._aenter[thread_id] is not None

    @classmethod
    def _increment_ref_count(cls, *, thread_id: int):
        assert cls._class_is_created_or_raise()
        if thread_id not in cls._ref_count:
            cls._ref_count[thread_id] = 0
        cls._ref_count[thread_id] = cls._ref_count[thread_id] + 1

    @classmethod
    def _decrement_ref_count(cls, *, thread_id: int):
        assert cls._class_is_created_or_raise()
        cls._ref_count[thread_id] = cls._ref_count[thread_id] - 1

    @classmethod
    def _cleanup(cls, *, thread_id: int):
        assert cls._class_is_created_or_raise()
        cls._shared_client[thread_id] = None
        cls._aenter[thread_id] = None

    @classmethod
    async def get_context_client(cls) -> aiohttp.ClientSession:
        assert cls._class_is_created_or_raise()
        async with PersistentClientSession() as client:
            return client

    # @classmethod
    # async def get_initialized_client(cls) -> Optional[aiohttp.ClientSession]:
    #     assert cls._class_is_created_or_raise()
    #     thread_id = threading.get_ident()
    #     if thread_id in cls._shared_client:
    #         return cls._shared_client[thread_id]
    #     return None
