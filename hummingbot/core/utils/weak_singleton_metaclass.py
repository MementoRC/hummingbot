import abc
import asyncio
import functools
import threading
import weakref
from typing import Dict, TypeVar

from typing_extensions import Type


class ClassNotYetInstantiatedError(Exception):
    """Exception raised when the WeakSingletonMetaclass object is not instantiated."""
    pass


class ClassAlreadyInstantiatedError(Exception):
    """Exception raised when the WeakSingletonMetaclass object is not instantiated."""
    pass


T = TypeVar("T", bound="WeakSingletonMetaclass")


class WeakSingletonMetaclass(abc.ABCMeta):
    """Metaclass for implementing a weak reference to singleton class.

    The WeakSingletonMetaclass metaclass ensures that only one instance of a class
    can exist at any time. It uses a dictionary to store the instances of each class.
    """

    _instances: Dict[Type[T], weakref.ref] = {}
    _locks: Dict[Type[T], type(threading.RLock())] = {}

    def __call__(cls: Type[T], *args, **kwargs):
        if cls not in cls._instances or cls._instances[cls]() is None:
            if cls not in cls._locks:
                cls._locks[cls] = threading.RLock()
            with cls._locks[cls]:
                cls.__cleanup_if_no_reference(cls)
                if cls not in cls._instances:
                    instance = super(WeakSingletonMetaclass, cls).__call__(*args, **kwargs)
                    cls._instances[cls] = weakref.ref(instance, functools.partial(cls.__cleanup_if_no_reference, cls))
                    return instance
        # I could do this:
        # if hasattr(cls._instances[cls](), "__post_call"):
        #     cls._instances[cls]().__getattribute__("__post_call")()
        return cls._instances[cls]()

    @classmethod
    def is_class_instantiated(mcs, cls: Type[T]) -> bool:
        return cls in mcs._instances and mcs._instances[cls]() is not None

    # --- Metaclass private methods ---
    # These methods are not intended to be used outside of this metaclass
    # They use "cls.xxx" to access the metaclass attributes xxx from the metaclass
    # It is a bit confusing: cls._instances[cls] -> WeakSingleInstanceClassMeta._instances[cls]
    # where cls is the Class that uses this metaclass
    def _clear_to_non_instantiated(cls: Type[T]):
        """Clear the class attributes
        This method should be implemented by the Class using this metaclass to do any
        necessary cleanup before the class is cleared."""
        pass

    async def _async_clear_to_non_instantiated(cls: Type[T]):
        """Asynchronously clears the class attributes
        This method should be implemented by the Class using this metaclass to do any
        necessary asynchronous cleanup before the class is cleared."""
        pass

    @classmethod
    def __cleanup_if_no_reference(mcs, cls: Type[T], msg: str = None):
        """Cleanup the references to the class if its weak reference is dead (last instance deleted)"""
        if cls in mcs._instances and mcs._instances[cls]() is None:
            with mcs._locks[cls]:
                if cls in mcs._instances and mcs._instances[cls]() is None:
                    mcs._instances.pop(cls)
                    mcs._locks.pop(cls)

    # --- Class testing methods ---
    def __clear(cls: Type[T]):
        """Clear the class attributes (registration of the class as Singleton)
        This method was added for testing purposes only. It should not be used in production code.
        This only simulates the class not being referenced as we cannot delete the instance directly."""
        if cls in cls._instances:
            with cls._locks[cls]:
                cls.__cleanup_if_no_reference(cls)
                if cls in cls._instances:
                    # Execute the clear method from the subclass
                    cls._instances[cls]().__class__._clear_to_non_instantiated()

                    setattr(cls, "__singleton_class_instantiated", False)
                    del cls._instances[cls]
                    del cls._locks[cls]

    async def __aclear(cls: Type[T]):
        """Clear the class attributes (registration of the class as Singleton)
        This method was added for testing purposes only. It should not be used in production code."""
        if cls in cls._instances and cls._instances[cls]() is not None:
            with cls._locks[cls]:
                if cls in cls._instances and cls._instances[cls]() is not None:
                    # Execute the clear method from the subclass
                    await cls._instances[cls]().__class__._async_clear_to_non_instantiated()

                    setattr(cls, "__singleton_class_instantiated", False)
                    del cls._instances[cls]
                    del cls._locks[cls]
        await asyncio.sleep(0)

    def __lock(cls: Type[T]):
        """Returns the lock for the class.
        This method was added for testing purposes only. It should not be used in production code."""
        if cls in cls._instances:
            return cls._locks[cls]
        else:
            raise ClassNotYetInstantiatedError(f"Class {cls.__name__} not yet instantiated. Nothing to lock")
