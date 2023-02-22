import asyncio
import ctypes
import sys
import threading
import unittest
import weakref
from unittest.mock import AsyncMock, patch

import aiohttp

from hummingbot.core.web_assistant.connections.persistent_client_session import (
    ClassNotYetInstantiatedError,
    NotWithinAsyncFrameworkError,
    PersistentClientSession,
)


class AsyncContextManagerMock(AsyncMock):
    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


class TestPersistentClientSession(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.PersistentClientSession = PersistentClientSession
        if self.PersistentClientSession.is_class_instantiated(self.PersistentClientSession):
            self.PersistentClientSession.__class__._SingleInstanceClassMeta__clear(self.PersistentClientSession)

    async def asyncTearDown(self):
        if self.PersistentClientSession.is_class_instantiated(self.PersistentClientSession):
            await self.PersistentClientSession.__class__._SingleInstanceClassMeta__aclear(self.PersistentClientSession)

    def tearDown(self):
        if self.PersistentClientSession.is_class_instantiated(self.PersistentClientSession):
            self.PersistentClientSession.__class__._SingleInstanceClassMeta__clear(self.PersistentClientSession)

    def test__init__raises_without_loop(self):
        with self.assertRaises(NotWithinAsyncFrameworkError):
            self.PersistentClientSession()

    async def test__init__create_with_async(self):
        instance = self.PersistentClientSession()
        print(instance)
        self.PersistentClientSession()
        print(instance)
        await asyncio.sleep(0)

    async def test__aenter__create_async_with_instance_no_as(self):
        instance = self.PersistentClientSession()
        async with instance:
            self.assertTrue(instance is self.PersistentClientSession())
        # In this case, there is no way to identify that the instance was created outside the async context manager
        self.assertFalse(instance._is_live_session(thread_id=threading.get_ident()))
        await asyncio.sleep(0)

    async def test__aenter__create_async_with_instance_as(self):
        instance = self.PersistentClientSession()
        async with instance as a:
            self.assertTrue(instance is self.PersistentClientSession())
            self.assertIsInstance(a, aiohttp.ClientSession)
        self.assertTrue(instance._is_live_session(thread_id=threading.get_ident()))
        await asyncio.sleep(0)

    async def test__aenter__create_with_async_0(self):
        print("Here")
        async with self.PersistentClientSession():
            print("There")
        await asyncio.sleep(0)

    async def test__aenter__create_with_async_1(self):
        async with (a := self.PersistentClientSession()):
            a
        await asyncio.sleep(0)

    def test_singleton_is_always_same_object(self):
        assert PersistentClientSession() is PersistentClientSession()

        # Sanity check - a non-singleton class should create two separate
        #  instances
        class NonSingleton:
            pass

        assert NonSingleton() is not NonSingleton()

    def test_assignment_of_instance_attr_raises(self):
        assert PersistentClientSession() is PersistentClientSession()

        # Sanity check - a non-singleton class should create two separate
        #  instances
        class NonSingleton:
            pass

        assert NonSingleton() is not NonSingleton()

    def test_class_created_does_not_raise(self):
        try:
            # code that creates the class
            my_class = PersistentClientSession()
        except ClassNotYetInstantiatedError:
            self.fail("ClassNotYetInstantiatedError raised unexpectedly")
        else:
            self.assertIsInstance(my_class, PersistentClientSession)

    def test_create_instance_raises_outside_event_loop(self):
        with self.assertRaises(NotWithinAsyncFrameworkError):
            self.PersistentClientSession()

    async def test_async_create_instance(self):
        instance = self.PersistentClientSession()
        self.assertIsInstance(instance, PersistentClientSession)
        self.assertIsInstance(instance._session_mutex, asyncio.Lock)
        self.assertIsInstance(instance._shared_client_session[threading.get_ident()], aiohttp.ClientSession)
        self.assertEqual(instance._context_ref_count[threading.get_ident()], 1)

    def test_class_not_created_error(self):
        # Test concurrency may have created the instance: Need to lock and clear to effectively test
        # However, if no instance exists, the lock cannot be acquired: Create the instance
        class LocalPersistentClientSession(PersistentClientSession):
            pass

        with self.assertRaises(ClassNotYetInstantiatedError):
            LocalPersistentClientSession._class_is_created_or_raise()

    def test_increment_class_not_created_error(self):
        with self.assertRaises(ClassNotYetInstantiatedError):
            PersistentClientSession._increment_context_ref_count(thread_id=threading.get_ident())

    def test_decrement_class_not_created_error(self):
        with self.assertRaises(ClassNotYetInstantiatedError):
            PersistentClientSession._decrement_context_ref_count(thread_id=threading.get_ident())

    def test_cleanup_session_class_not_created_error(self):
        with self.assertRaises(ClassNotYetInstantiatedError):
            PersistentClientSession._cleanup_session_session(thread_id=threading.get_ident())

    async def test__aenter_class_not_instantiated_error(self):
        with self.assertRaises(ClassNotYetInstantiatedError):
            await PersistentClientSession.__aenter__()

    async def test__aexit_class_not_instantiated_error(self):
        with self.assertRaises(ClassNotYetInstantiatedError):
            await PersistentClientSession.__aexit__(*sys.exc_info())

    async def test__create_session_class_not_created_error(self):
        with self.assertRaises(ClassNotYetInstantiatedError):
            await PersistentClientSession.__async_create_session(thread_id=threading.get_ident())

    async def test__cleanup_session_class_not_created_error(self):
        with self.assertRaises(ClassNotYetInstantiatedError):
            await PersistentClientSession._cleanup_closed_session(thread_id=threading.get_ident())

    async def test_get_context_client_class_not_created_error(self):
        with self.assertRaises(ClassNotYetInstantiatedError):
            await PersistentClientSession.async_get_session()

    @patch("aiohttp.ClientSession")
    async def test__create_session_failure(self, mock_client_session):
        mock_client_session.side_effect = Exception("error creating session")
        instance = PersistentClientSession()

        with self.assertRaises(Exception):
            await instance._async_create_session(thread_id=threading.get_ident())

    @patch("aiohttp.ClientSession")
    async def test__create_session(self, mock_client_session):
        mock_client_session.side_effect = Exception("error creating session")
        instance = PersistentClientSession()

        with self.assertRaises(Exception):
            await instance._async_create_session(thread_id=threading.get_ident())

    async def test_aenter(self):
        async with PersistentClientSession() as client:
            self.assertEqual(False, client.closed)
            await asyncio.sleep(0.1)
        self.assertEqual(True, client.closed)
        await asyncio.sleep(0.1)

    async def test_aenter_with_instance(self):
        instance = PersistentClientSession()
        async with instance as client:
            self.assertEqual(False, client.closed)
            await asyncio.sleep(0.1)
        self.assertEqual(False, client.closed)
        await asyncio.sleep(0.1)

    async def test_aenter_with_instance_and_class(self):
        instance = PersistentClientSession()
        async with PersistentClientSession() as client:
            self.assertEqual(False, client.closed)
            await asyncio.sleep(0.1)
        self.assertEqual(False, client.closed)
        del instance
        self.assertEqual(True, client.closed)
        await asyncio.sleep(0.1)

    async def test_aenter_with_return(self):
        async def return_client():
            async with PersistentClientSession() as client:
                self.assertEqual(False, client.closed)
                await asyncio.sleep(0.1)
                return client

        client = await return_client()
        # Session is not closed, deferred to user close
        self.assertEqual(False, client.closed)
        await client.close()
        self.assertEqual(True, client.closed)

        client = await return_client()
        # Session is not closed, deferred to user close
        self.assertEqual(False, client.closed)
        del client
        print(PersistentClientSession.has_live_session(thread_id=threading.get_ident()))
        await asyncio.sleep(0.1)

    async def test_aexit_not_called_on_outcontext(self):
        with patch("aiohttp.ClientSession") as mock_client_session:
            instance = PersistentClientSession()
            mock_client_session.return_value = AsyncContextManagerMock()
            async with instance:
                await asyncio.sleep(0.1)

            self.assertFalse(mock_client_session.return_value.__aexit__.called)
            self.assertEqual(mock_client_session.return_value.__aexit__.await_count, 0)

    async def test_async_get_session(self):
        instance = PersistentClientSession()
        thread_id = threading.get_ident()
        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_client_session.return_value = AsyncContextManagerMock(return_value="client")

            # session = await next(instance.async_yield_session_in_context())
            # print(f"session: {sys.getrefcount(instance._shared_session[thread_id])}")
            client = await instance.async_get_session()
            print(f"session: {client} {sys.getrefcount(client)} {weakref.getweakrefcount(client)}")
            self.assertTrue(mock_client_session.return_value.__aenter__.called)
            self.assertEqual(mock_client_session.return_value.__aenter__.await_count, 0)
            self.assertTrue(type(client), type(aiohttp.ClientSession))
            self.assertTrue(client, mock_client_session.return_value)
            self.assertEqual(await client, await mock_client_session.return_value.__aenter__())
            self.assertEqual(mock_client_session.return_value.__aenter__.await_count, 2)

            # __aenter__ is called twice: Once when the context manager is created, and once when it is asserted
            # self.assertTrue(mock_client_session.return_value.__aexit__.called)
            # self.assertEqual(mock_client_session.return_value.__aexit__.await_count, 1)

            # PersistentClientSession counter keeps track of the number of sessions
            self.assertNotEqual(None, (instance._shared_client_session[thread_id]))
            # PersistentClientSession in-of-context state is empty
            # self.assertTrue(thread_id not in instance._context_ref_count)
            # self.assertTrue(thread_id in instance._all_ref_count)

            # PersistentClientSession out-of-context state is non-empty
            # self.assertTrue(instance._finalizer[thread_id].alive)
            client = None
            print(f"session: {client} {sys.getrefcount(client)} {weakref.getweakrefcount(client)}")
            print(f"session: {weakref.getweakrefcount(instance._shared_client_session[thread_id])}")
            # self.assertFalse(instance._finalizer[thread_id].alive)

    async def test_finalize_client(self):
        thread_id = threading.get_ident()
        instance = PersistentClientSession()

        client = await instance.async_get_session()
        print(f"{client} thread_id: {thread_id}")
        print(f"{instance._shared_client_session} thread_id: {thread_id}")
        # self.assertEqual(client, instance._shared_client_session[thread_id])
        del client
        # client = await instance.async_get_client_in_context()

    # async def test_async_get_context_client(self):
    #    instance = PersistentClientSession()
    #    with patch("aiohttp.ClientSession") as mock_client_session:
    #        mock_client_session.return_value = AsyncContextManagerMock(return_value="client")


#
#        client = await instance.async_get_client_in_context()
#        self.assertTrue(client, mock_client_session.return_value)
#        del client
#        print("client deleted")

class TestA:
    ref_count = 0
    _ref_count = 0
    client: aiohttp.ClientSession
    weakref_client: weakref.ref = None
    proxy_client: weakref.proxy = None
    proxy_ref_count = 0
    proxy_wea_count = 0
    weakref_ref_count = 0
    weakref_wea_count = 0
    true_ref_count = 0

    __slots__ = ("client",)

    @staticmethod
    def update_ref_count(prefix: str = ""):
        if TestA.client is not None:
            TestA.true_ref_count = ctypes.c_long.from_address(id(TestA.client)).value
        else:
            TestA.true_ref_count = 0
        if TestA.weakref_client is not None and TestA.weakref_client() is not None:
            TestA.weakref_ref_count = ctypes.c_long.from_address(id(TestA.weakref_client())).value
            TestA.weakref_wea_count = weakref.getweakrefcount(TestA.weakref_client())
        else:
            TestA.weakref_ref_count = 0
            TestA.weakref_wea_count = 0
        if TestA.proxy_client is not None:
            TestA.proxy_ref_count = ctypes.c_long.from_address(id(TestA.proxy_client)).value
            TestA.proxy_wea_count = weakref.getweakrefcount(TestA.proxy_client)
        else:
            TestA.proxy_ref_count = 0
            TestA.proxy_wea_count = 0
        print(
            f"{prefix:35} {'count':>10}:{'true':>10}:{'true (wr)':>10}:{'weakref':>10}:{'true (pr)':>10}:{'WR (pr)':>10}")
        print(
            f"{' ':35} {TestA.ref_count:10}:{TestA.true_ref_count:10}:{TestA.weakref_ref_count:10}:{TestA.weakref_wea_count:10}:{TestA.proxy_ref_count:10}:{TestA.proxy_wea_count:10}")

    async def __aenter__(self, **kwargs):
        TestA.ref_count += 1

        TestA.client = aiohttp.ClientSession(**kwargs)
        weakref.finalize(TestA.client, TestA.close_session, aiohttp.ClientSession.close)

        TestA.update_ref_count("__aenter__ Session")
        TestA.weakref_client = weakref.ref(TestA.client)
        TestA.update_ref_count("__aenter__ WeakRef")
        TestA.proxy_client = weakref.proxy(TestA.client)
        TestA.update_ref_count("__aenter__ Proxy")
        TestA._ref_count = TestA.weakref_ref_count
        print(f"   __aenter__: {TestA.weakref_ref_count}")
        return TestA.client.__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        out_context_ref_count = TestA.weakref_ref_count
        print(f"   __aexit__: {out_context_ref_count}")
        TestA.update_ref_count("__aexit__ entering")
        print(f"   __aexit__: {out_context_ref_count}")
        out_context_ref_count = TestA.weakref_ref_count - TestA._ref_count

        TestA.ref_count = out_context_ref_count - 1
        if TestA.ref_count == 0 and TestA.true_ref_count == 1:
            TestA.update_ref_count("__aexit__ closing ClientSession")
            await TestA.client.close()
        else:
            # Set a finalizer on the pending reference and terminate the current reference
            TestA.update_ref_count("__aexit__ before deleting client")
            TestA.client = None
            TestA.update_ref_count("__aexit__ after deleting client")

    @classmethod
    def close_session(cls, close):
        TestA.ref_count -= 1
        TestA.update_ref_count("close_session")
        print(f"close_session: {TestA.weakref_client().closed}")
        # asyncio.ensure_future(close(TestA.weakref_client()))
        # await TestA.weakref_client.close()


class TestTestA(unittest.IsolatedAsyncioTestCase):
    async def test_session_closes_when_no_references(self):
        async def get_client():
            async with TestA() as client:
                TestA.update_ref_count("get_client inside with")
                return client

        ret = await get_client()
        TestA.update_ref_count("returned from get_client")
        self.assertTrue(ret.closed)
        del ret
        TestA.update_ref_count("after del ret")

    async def test_session_closes_with_yielded_reference(self):
        async def yield_client():
            async with TestA() as client:
                TestA.update_ref_count("         yield_client inside with")
                yield client
                TestA.update_ref_count("         yield_client after yield")
            TestA.update_ref_count("   yield_client outside with")

        out_ref: aiohttp.ClientSession
        async for y in yield_client():
            out_ref = y
            self.assertFalse(y.closed)
            TestA.update_ref_count("yield_client outside with")

        TestA.update_ref_count("only one reference to client")

        self.assertFalse(out_ref.closed)
        out_ref = None
        TestA.update_ref_count("only one reference to client")
        self.assertFalse(y.closed)
        await asyncio.sleep(1)
        y = None
        print("Client:", TestA.weakref_client())
        # self.assertTrue(TestA.weakref_client() is None)

    async def test_multiple_references(self):
        async with TestA() as client1:
            async with TestA() as client2:
                pass

            self.assertFalse(client1.closed)
            self.assertTrue(client2.closed)
