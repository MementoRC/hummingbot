# distutils: language=c++
# TimeIterator is now pure Python (C5 conversion); cimport removed.
# PyTimeIterator is kept as a cdef class that inherits from object, holding
# a TimeIterator reference is no longer possible via cdef inheritance.
# Revert to a plain Python subclass — the .pyx wrapping is a no-op here.

from hummingbot.core.time_iterator import TimeIterator


class PyTimeIterator(TimeIterator):
    def tick(self, timestamp: float) -> None:
        super().tick(timestamp)
