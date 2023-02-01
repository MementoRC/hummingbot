import asyncio
import threading
from typing import Any, Coroutine, Dict, Optional, Union

import aiohttp


class ClassNotYetCreatedError(Exception):
    pass


class PersistentClientSession:
    """ A persistent aiohttp.ClientSession that can be shared across threads and asyncio tasks."""
    __instance: Optional["PersistentClientSession"] = None
    _lock = threading.Lock()

    _session_mutex = asyncio.Lock()
    _shared_client: Dict[int, Union[aiohttp.ClientSession, None]] = dict()
    _ref_count: Dict[int, int] = dict()

    __slots__ = ()

    def __new__(cls):
        if cls.__instance is None:
            with cls._lock:
                # Another thread could have created the instance
                # before we acquired the lock. So check that the
                # instance is still nonexistent.
                if not cls.__instance:
                    cls.__instance = super().__new__(cls)
        return cls.__instance

    @classmethod
    async def __aenter__(cls, **kwargs) -> Coroutine[Any, Any, aiohttp.ClientSession]:
        assert cls._class_is_created_or_raise()
        thread_id = threading.get_ident()
        # Increment the ref count for the current thread
        cls._increment_ref_count(thread_id=thread_id)
        await cls._create_session(thread_id=thread_id, **kwargs)
        return cls._shared_client[thread_id].__aenter__()

    @classmethod
    async def __aexit__(cls, exc_type, exc_val, exc_tb):
        assert cls._class_is_created_or_raise()
        thread_id = threading.get_ident()
        # Decrement the ref count for the current thread
        cls._ref_count[thread_id] = cls._ref_count[thread_id] - 1

        # If the ref count is 0, we clean up
        if cls._ref_count[thread_id] == 0:
            # If the ref count is 0, we can safely close the session
            await cls._shared_client[thread_id].__aexit__(None, None, None)
            cls._cleanup(thread_id=thread_id)
        await asyncio.sleep(0)

    @classmethod
    def _class_is_created_or_raise(cls):
        if cls.__instance is None:
            raise ClassNotYetCreatedError("Class not created yet."
                                          f"The methods of `{cls.__class__.__name__}' can only be called after this 'Singleton' class has been created.")
        return True

    @classmethod
    async def _create_session(cls, *, thread_id: int, **kwargs):
        assert cls._class_is_created_or_raise()
        # If the session is already created, we can return
        if thread_id in cls._shared_client and cls._shared_client[thread_id] is not None:
            await asyncio.sleep(0)
            return

        # Otherwise, we need to create the session
        async with cls._session_mutex:
            # Another thread could have created the session
            # before we acquired the lock. So check that the
            # session is still nonexistent.
            if thread_id in cls._shared_client and cls._shared_client[thread_id] is not None:
                await asyncio.sleep(0)
                return

            try:
                cls._shared_client[thread_id] = aiohttp.ClientSession(**kwargs)
            except RuntimeError as e:
                cls._shared_client[thread_id] = None
                print(f"Error creating aiohttp session: {e}")
                raise e

            await asyncio.sleep(0)
        assert cls._shared_client[thread_id] is not None

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
        cls._ref_count[thread_id] = 0

    @classmethod
    def __clear(cls):
        PersistentClientSession.__instance = None

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
