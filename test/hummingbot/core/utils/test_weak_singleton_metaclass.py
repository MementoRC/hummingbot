import random
import threading
import unittest
from time import sleep
from unittest.mock import AsyncMock

from hummingbot.core.utils.weak_singleton_metaclass import ClassNotYetInstantiatedError, WeakSingletonMetaclass


class AsyncContextManagerMock(AsyncMock):
    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


class TestWeakSingletonMetaclass(unittest.TestCase):
    def setUp(self):
        class SingletonClass(metaclass=WeakSingletonMetaclass):
            def __init__(self, val):
                self.val = val

        class LocalThread(threading.Thread):
            def __init__(self, sleep: float = 0, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.result = None
                self.sleep = sleep

            def run(self):
                instance = SingletonClass(random.randint(0, 10000))
                self.result = instance.val
                sleep(self.sleep)

        self.SingletonClass = SingletonClass
        self.LocalThread = LocalThread

    def test_multi_thread_behavior_with_first_instance_pass(self):
        a = self.SingletonClass(10)
        threads = [self.LocalThread() for i in range(1000)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        results = [t.result for t in threads]
        self.assertTrue(all(
            result == a.val for result in results), f"Expected all results to be {results[0]}, but got {results}")

    def test_multi_thread_behavior_without_first_instance_fail(self):
        # Check that a singleton created by one thread dies at the end of the thread (due to only having w weak reference)
        # and that a new instance is created by another thread (which will have a different memory address and value)
        threads = [self.LocalThread(0) for i in range(1000)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        results = [t.result for t in threads]

        # Fuzzy test: Some results may be the same as the first one, but most should be different
        # The first thread creates a random number and a few will see the same value before the thread dies
        # There could be some chaining, where the first thread dies is kept alive by the second thread, etc.
        self.assertFalse(any(result == results[0] for result in results[100:]),
                         f"Expected all results to be {results[0]}, but got {results}")
        # All remaining result should be different from the first one
        self.assertTrue(all(result != results[0] for result in results[100:]),
                        f"Expected all results to be {results[0]}, but got {results}")

    def test_singleton_behavior(self):
        # Check that two instances of the same class have the same memory address
        #
        a = self.SingletonClass(10)
        b = self.SingletonClass(20)
        self.assertIs(a, b)
        self.assertEqual(a.val, 10)
        self.assertEqual(b.val, 10)

        # Check that a singleton created by one thread is accessible by another
        result = [None]

        def set_singleton_value():
            singleton = self.SingletonClass(30)
            result[0] = singleton.val

        thread = threading.Thread(target=set_singleton_value)
        thread.start()
        thread.join()
        self.assertEqual(10, result[0])
        self.assertEqual(a.val, 10)

    def test_weak_singleton_behavior(self):
        def set_singleton_value():
            singleton = self.SingletonClass(30)
            result[0] = singleton.val

        # Check that two instances of the same class have the same memory address
        a = self.SingletonClass(10)
        b = self.SingletonClass(20)
        self.assertIs(a, b)
        self.assertEqual(a.val, 10)
        self.assertEqual(b.val, 10)

        # Remove all hard-reference: This will automatically delete the instance
        # since no hard-references are stored inside the WeakSingletonMetaclass
        del a, b

        # Since no instance exists, a thread can create the instance
        result = [None]
        thread = threading.Thread(target=set_singleton_value)
        thread.start()
        thread.join()
        self.assertEqual(30, result[0])

    def test_subclass_behavior(self):
        # Check that subclasses of a singleton class are different singletons
        class SubclassA(self.SingletonClass):
            pass

        class SubclassB(self.SingletonClass):
            pass

        a = SubclassA(10)
        b = SubclassB(20)
        self.assertIsNot(a, b)

        c = SubclassA(30)
        self.assertIs(a, c)

    def test__class_is_instantiated(self):
        # Check that subclasses of a singleton class are different singletons
        class ClassA(metaclass=WeakSingletonMetaclass):
            pass

        class SubclassA(ClassA):
            pass

        self.assertFalse(ClassA.is_instantiated(ClassA))
        self.assertFalse(SubclassA.is_instantiated(SubclassA))

        a = ClassA()
        self.assertTrue(ClassA.is_instantiated(ClassA))
        self.assertFalse(SubclassA.is_instantiated(SubclassA))
        self.assertTrue(a.__class__.is_instantiated(a.__class__))

        b = SubclassA()
        self.assertTrue(ClassA.is_instantiated(ClassA))
        self.assertTrue(SubclassA.is_instantiated(SubclassA))
        self.assertTrue(a.__class__.is_instantiated(a.__class__))
        self.assertTrue(b.__class__.is_instantiated(b.__class__))

    def test__lock_raises(self):
        # Check that subclasses of a singleton class are different singletons'
        with self.assertRaises(ClassNotYetInstantiatedError):
            with self.SingletonClass.__class__._WeakSingletonMetaclass__lock(self.SingletonClass):
                pass

    def test__lock_reentrant(self):
        # Check that subclasses of a singleton class are different singletons
        a = self.SingletonClass(0)
        self.assertTrue(self.SingletonClass.is_instantiated(self.SingletonClass))

        # This method is implemented only for test purposes, the lock is not supposed to be used directly
        lock = self.SingletonClass.__class__._WeakSingletonMetaclass__lock(self.SingletonClass)
        with lock:
            # This only intend to use several methods that require using the Lock() object
            self.assertTrue(self.SingletonClass.is_instantiated(self.SingletonClass))

            # Re-entrant lock allows to __clear() the singleton from the same thread
            self.SingletonClass.__class__._WeakSingletonMetaclass__clear(self.SingletonClass)
            self.assertFalse(self.SingletonClass.is_instantiated(self.SingletonClass))

            # A cleared singleton can be re-instantiated (because the reference to the current lock is deleted)
            b = self.SingletonClass(0)
            self.assertTrue(self.SingletonClass.is_instantiated(self.SingletonClass))
            self.assertIsNot(a, b)

    def test__lock_reentrant_thread(self):
        class SingletonClass(metaclass=WeakSingletonMetaclass):
            def __init__(self, val):
                self.val = val

        class LockingThread(threading.Thread):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.result = None

            def run(self):
                SingletonClass(0)
                if SingletonClass.__class__._WeakSingletonMetaclass__lock(SingletonClass).acquire(False):
                    self.result = "Acquired"
                else:
                    self.result = "Not acquired"
                sleep(0.1)

        # Check that subclasses of a singleton class are different singletons
        a = SingletonClass(0)
        self.assertTrue(a is SingletonClass(0))
        self.assertTrue(SingletonClass.is_instantiated(SingletonClass))
        lock = SingletonClass.__class__._WeakSingletonMetaclass__lock(SingletonClass)
        with lock:
            # Threads are not re-entrant
            t = LockingThread()
            t.start()
            t.join()
            self.assertEqual(t.result, "Not acquired")

    def test__lock_reentrant_thread_on_dead_instance(self):
        class SingletonClass(metaclass=WeakSingletonMetaclass):
            def __init__(self, val):
                self.val = val

        class LockingThread(threading.Thread):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.result = None

            def run(self):
                a = SingletonClass(0)
                assert a is SingletonClass(0)
                if SingletonClass.__class__._WeakSingletonMetaclass__lock(SingletonClass).acquire(False):
                    self.result = "Acquired"
                else:
                    self.result = "Not acquired"
                sleep(0.1)

        # Check that subclasses of a singleton class are different singletons
        a = SingletonClass(0)
        self.assertTrue(SingletonClass.is_instantiated(SingletonClass))
        lock = SingletonClass.__class__._WeakSingletonMetaclass__lock(SingletonClass)
        with lock:
            del a
            self.assertFalse(SingletonClass.is_instantiated(SingletonClass))

            # Threads are not re-entrant
            t = LockingThread()
            t.start()
            t.join()
            self.assertEqual(t.result, "Acquired")

    def test_clear_singleton_from_class(self):
        self.assertFalse(self.SingletonClass.is_instantiated(self.SingletonClass))

        a = self.SingletonClass(0)
        self.assertTrue(a is self.SingletonClass(0))
        self.assertTrue(self.SingletonClass.is_instantiated(self.SingletonClass))
        # Access to a private method to clear the singleton
        self.SingletonClass._clear_to_non_instantiated()
        self.assertTrue(self.SingletonClass.is_instantiated(self.SingletonClass))

    def test__clear_singleton_from_class(self):
        # A hard reference to the singleton needs to exist, otherwise the WeakSingleton vanishes
        a = self.SingletonClass(0)
        self.assertTrue(a is self.SingletonClass(0))
        self.assertIn(self.SingletonClass, WeakSingletonMetaclass._instances)
        self.assertIn(self.SingletonClass, WeakSingletonMetaclass._locks)
        self.SingletonClass.__class__._WeakSingletonMetaclass__clear(self.SingletonClass)
        self.assertNotIn(self.SingletonClass, WeakSingletonMetaclass._locks)
