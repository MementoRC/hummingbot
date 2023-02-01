import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, patch

from hummingbot.core.web_assistant.connections.persistent_client_session import (
    ClassNotYetCreatedError,
    PersistentClientSession,
)


class TestPersistentClientSession(unittest.IsolatedAsyncioTestCase):
    async def test_class_not_created_error(self):
        with PersistentClientSession()._lock:
            # Reset the singleton instance, since other tests may have created it
            PersistentClientSession._PersistentClientSession__instance = None
            with self.assertRaises(ClassNotYetCreatedError):
                PersistentClientSession._class_is_created_or_raise()

    def test_singleton_is_always_same_object(self):
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
        except ClassNotYetCreatedError:
            self.fail("ClassNotYetCreatedError raised unexpectedly")
        else:
            self.assertIsInstance(my_class, PersistentClientSession)

    def test_create_instance(self):
        with PersistentClientSession()._lock:
            # Reset the singleton instance, since other tests may have created it
            PersistentClientSession._PersistentClientSession__clear()

            instance = PersistentClientSession()
            self.assertIsInstance(instance, PersistentClientSession)
            self.assertEqual(type(instance._lock), type(threading.Lock()))
            self.assertIsInstance(instance._session_mutex, asyncio.Lock)
            self.assertEqual(instance._shared_client, dict())
            self.assertEqual(instance._ref_count, dict())

    @patch("aiohttp.ClientSession")
    async def test_create_session_failure(self, mock_client_session):
        mock_client_session.side_effect = Exception("error creating session")
        instance = PersistentClientSession()

        with self.assertRaises(Exception):
            await instance._create_session(thread_id=threading.get_ident())

    async def test_aenter(self):
        instance = PersistentClientSession()
        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_client_session.return_value.__aenter__ = AsyncMock(return_value="client")
            async with instance as client:
                self.assertEqual(await client, "client")

            self.assertTrue(mock_client_session.return_value.__aenter__.called)
            self.assertEqual(1, mock_client_session.return_value.__aenter__.await_count)

    async def test_aexit(self):
        instance = PersistentClientSession()
        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_client_session.return_value.__aenter__ = AsyncMock(return_value="client")
            mock_client_session.return_value.__aexit__ = AsyncMock()
            async with instance:
                pass

            self.assertTrue(mock_client_session.return_value.__aexit__.called)
            self.assertEqual(mock_client_session.return_value.__aexit__.await_count, 1)

    async def test_get_context_client(self):
        instance = PersistentClientSession()
        with patch("aiohttp.ClientSession") as mock_client_session:
            mock_client_session.return_value = "client"

            client = await instance.get_context_client()
            self.assertEqual(client, "client")
