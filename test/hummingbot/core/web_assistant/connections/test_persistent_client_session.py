import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import ClientSession

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
        self.client_type = PersistentClientSession
        if self.client_type.is_instantiated():
            self.client_type.__class__._WeakSingletonMetaclass__clear(self.client_type)

    async def asyncTearDown(self):
        await asyncio.sleep(0)

    def tearDown(self):
        if self.client_type.is_instantiated():
            self.client_type.__class__._WeakSingletonMetaclass__clear(self.client_type)

    def assertInstantiatedInThread(self, instance: PersistentClientSession, thread_id: int):
        self.assertTrue(instance.__class__.is_instantiated())
        self.assertIsInstance(instance, PersistentClientSession)
        self.assertTrue(thread_id in instance._shared_client_sessions)
        self.assertTrue(thread_id in instance._kwargs_client_sessions)
        self.assertTrue(thread_id in instance._sessions_mutex)
        self.assertTrue(thread_id in instance._in_context_count)
        self.assertTrue(thread_id in instance._ref_count)
        self.assertTrue(issubclass(asyncio.Lock, type(instance._sessions_mutex.get(thread_id)), ))
        self.assertIsNotNone(instance._sessions_mutex.get(thread_id), )

    def assertInitializedStateInThread(self, instance: PersistentClientSession, thread_id: int, kwargs=None):
        if kwargs is None:
            kwargs = {}
        self.assertInstantiatedInThread(instance, thread_id)
        self.assertEqual(0, instance._in_context_count.get(thread_id))
        self.assertEqual(0, instance._ref_count.get(thread_id))
        self.assertEqual(None, instance._shared_client_sessions.get(thread_id))
        self.assertEqual(kwargs, instance._kwargs_client_sessions.get(thread_id), )
        self.assertFalse(instance._sessions_mutex.get(thread_id).locked())

    def assertSessionOpenedInThread(self, instance: PersistentClientSession, thread_id: int, kwargs=None):
        if kwargs is None:
            kwargs = {}
        self.assertInstantiatedInThread(instance, thread_id)
        self.assertEqual(kwargs, instance._kwargs_client_sessions.get(thread_id), )
        self.assertNotEqual(None, instance._shared_client_sessions.get(thread_id))
        self.assertIsInstance(instance._shared_client_sessions.get(thread_id), ClientSession)
        self.assertFalse(instance._shared_client_sessions.get(thread_id).closed)

    def assertSessionClosedInThread(self, instance: PersistentClientSession, thread_id: int, kwargs=None):
        if kwargs is None:
            kwargs = {}
        self.assertInstantiatedInThread(instance, thread_id)
        self.assertEqual(kwargs, instance._kwargs_client_sessions.get(thread_id), )
        self.assertNotEqual(None, instance._shared_client_sessions.get(thread_id))
        self.assertIsInstance(instance._shared_client_sessions.get(thread_id), ClientSession)
        self.assertFalse(instance._shared_client_sessions.get(thread_id).closed)

    def test_instantiated_state(self):
        thread_id: int = threading.get_ident()
        kwargs: dict = {"key": "val"}
        instance: PersistentClientSession = PersistentClientSession(key="val")

        self.assertInstantiatedInThread(instance, thread_id)
        self.assertInitializedStateInThread(instance, thread_id, kwargs)
        self.assertEqual(kwargs, instance._kwargs_client_sessions.get(thread_id), )

    async def test_session_opened_state(self):
        thread_id: int = threading.get_ident()
        instance: PersistentClientSession = PersistentClientSession()

        self.assertInstantiatedInThread(instance, thread_id)
        session: ClientSession = instance()
        self.assertTrue(session is instance())
        self.assertSessionOpenedInThread(instance, thread_id, None)

    async def test_session_closed_state(self):
        thread_id: int = threading.get_ident()
        instance: PersistentClientSession = PersistentClientSession()

        self.assertInstantiatedInThread(instance, thread_id)
        session: ClientSession = instance()
        self.assertTrue(session is instance())
        self.assertSessionOpenedInThread(instance, thread_id, None)

    def test___call___raises_without_event_loop(self):
        thread_id: int = threading.get_ident()
        instance: PersistentClientSession = PersistentClientSession(key="val")
        self.assertTrue(instance.__class__.is_instantiated())
        self.assertIsInstance(instance, PersistentClientSession)
        self.assertTrue(instance is PersistentClientSession())

        # Instantiating the ClientSession using __call__ should raise the exception from ClientSession
        with self.assertRaises(Exception) as loop_error:
            asyncio.get_running_loop()
        self.assertTrue('no running event loop' in str(loop_error.exception))

        with self.assertRaises(NotWithinAsyncFrameworkError) as instance_error:
            instance()

        self.assertIsInstance(instance_error.exception, NotWithinAsyncFrameworkError)
        self.assertTrue('The event loop is not running' in str(instance_error.exception))
        self.assertTrue(issubclass(asyncio.Lock, type(instance._sessions_mutex.get(thread_id)), ))
        self.assertIsNotNone(instance._sessions_mutex.get(thread_id), )
        self.assertEqual(None, instance._shared_client_sessions.get(thread_id), )

    async def test___call___does_not_creates_without_reference(self):
        client_session: ClientSession = PersistentClientSession()()
        # Without hard-reference, the instance is created and deleted immediately
        self.assertFalse(PersistentClientSession.is_instantiated())
        # The session has been created and closed
        # we cannot delete it as a cleanup, since it has a reference (i.e. local 'client_session')
        self.assertIsInstance(client_session, ClientSession)
        self.assertTrue(client_session.closed)
        await asyncio.sleep(0)

    async def test___call___creates_with_reference(self):
        thread_id: int = threading.get_ident()
        instance: PersistentClientSession = PersistentClientSession()

        client_session: ClientSession = instance()

        self.assertIsInstance(client_session, ClientSession)
        self.assertFalse(client_session.closed)
        self.assertTrue(thread_id in instance._shared_client_sessions)
        self.assertEqual(client_session, instance._shared_client_sessions[thread_id])
        # Note that this test does not generate an `ClientSession unclosed' post-run error
        # This indicates that the session is being closed properly when 'instance' gets unreferenced
        # TODO: Find a way to test that the session is closed when the instance is unreferenced
        await asyncio.sleep(0)

    @patch("hummingbot.core.web_assistant.connections.persistent_client_session.ClientSession")
    async def test___call___propagates_clientsession_exception(self, mock_client_session):
        mock_client_session.return_value = Exception("error creating session")

        instance: PersistentClientSession = PersistentClientSession()
        self.assertIsInstance(instance, PersistentClientSession)
        self.assertTrue(instance is PersistentClientSession())

        # Instantiating the ClientSession using __call__ should raise the exception from ClientSession
        with self.assertRaises(Exception, msg="Exception not raised"):
            instance()

    async def test__aenter__create_async_with_instance_no_as(self):
        instance = PersistentClientSession()
        async with instance:
            self.assertTrue(instance is self.PersistentClientSession())
        # In this case, there is no way to identify that the instance was created outside the async context manager
        self.assertFalse(instance.has_live_session(thread_id=threading.get_ident()))
        await asyncio.sleep(0)

    async def test__aenter__create_async_with_instance_as(self):
        instance = self.PersistentClientSession()
        async with instance as a:
            self.assertTrue(instance is self.PersistentClientSession())
            self.assertIsInstance(a, ClientSession)
        self.assertTrue(instance.has_live_session(thread_id=threading.get_ident()))
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
        self.assertIsInstance(instance._sessions_mutex, asyncio.Lock)
        self.assertIsInstance(instance._shared_client_sessions[threading.get_ident()], ClientSession)
        self.assertEqual(instance._context_ref_count[threading.get_ident()], 1)

    def test_class_not_created_error(self):
        # Test concurrency may have created the instance: Need to lock and clear to effectively test
        # However, if no instance exists, the lock cannot be acquired: Create the instance
        class LocalPersistentClientSession(PersistentClientSession):
            pass

        with self.assertRaises(ClassNotYetInstantiatedError):
            LocalPersistentClientSession.is_instantiated_or_raise()

    def test_cleanup_session_class_not_created_error(self):
        with self.assertRaises(ClassNotYetInstantiatedError):
            PersistentClientSession._cleanup_session_session(thread_id=threading.get_ident())

    @patch("aiohttp.ClientSession")
    async def test___call___propagates_exception(self, mock_client_session):
        mock_client_session.side_effect = Exception("error creating session")
        instance = PersistentClientSession()
        self.assertIsInstance(instance, PersistentClientSession)
        self.assertTrue(instance is PersistentClientSession())

        # Instantiating the ClientSession using __call__ should raise the exception from ClientSession
        with self.assertRaises(Exception):
            instance()

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
