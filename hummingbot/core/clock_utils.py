"""Provides a set of functions used primarily by the Clock and IteratorsThreadTimer classes.

Here are some of the useful functions provided by this module:

    s_ns() Converts a number in unit into its nanounit representation.
    ns_s() Converts a number in nanounit into its unit representation.
    tick_formula_ns() Rounds a given number into its closest multiple of a period.
    get_current_tick_s()
    get_current_tick_ns()
    @try_except Decorator wrapping a function with Exception handling
    @try_except_async Decorator wrapping an async function with Exception handling
    @in_executor Decorator wrapping function with asyncio executor with Exception handling
"""

# This module is in the public domain.  No warranties.

from __future__ import annotations

__author__ = ('MementoMori',
              'MementoRC')

import asyncio
import functools
import time
from decimal import Decimal
from typing import Callable, Union


def ns_s(nanounit: Union[int, str]) -> float:
    """
    Converts a number in nanounit into its unit representation.
    Uses Decimal as conversion from int|str to float.

    :param Union[int, str] nanounit: number in nanounit
    :returns: number in unit
    :rtype: float
    """
    return float(Decimal(nanounit) * Decimal("1e-9"))


def s_ns(unit: Union[float, str]) -> int:
    """
    Converts a number in unit into its nanounit representation.
    Uses Decimal as conversion from float|str to int.

    :param Union[float, str] unit: Number in unit
    :returns: Number in nanounit
    :rtype: int
    """
    return int(Decimal(unit) * Decimal("1e9"))


def tick_formula_ns(number: int, period: int, upper: bool = False) -> int:
    """
    Returns closest multiple of a given period to a number.
    Argument 'upper' selects whether to return the largest multiple smaller
    or the smallest multiple larger than the input

    :param int number: Number to be periodized
    :param int period: Period
    :param bool upper: Optional select of next larger or smaller
    :returns: Closest multiple
    :rtype: int
    """
    try:
        d = (number // period + int(upper)) * period
    except ZeroDivisionError:
        d = 0
    return d


def get_current_tick_s(period_s: Union[float | str]) -> float:
    """
    Returns the closest second multiple of a period to the current time (time.time_ns())

    :param Union[float | str] period_s: Period in second
    :returns: Closest multiple of the period to the current time
    :rtype: float
    """
    return ns_s(tick_formula_ns(time.time_ns(), s_ns(period_s)))


def get_current_tick_ns(period_ns: int) -> int:
    """
    Returns the closest nanosecond multiple of a period to the current time (time.time_ns())

    :param int period_ns: Period in nanosecond
    :returns: Closest multiple of the period to the current time
    :rtype: int
    """
    return tick_formula_ns(time.time_ns(), period_ns)


async def try_except_async(function: Callable):
    """
    Decorator wrapping an async function with exception handling

    :param Callable function: Period in nanosecond
    """
    async def call(*args, **kwargs):
        returned_value = None
        try:
            returned_value = await function(*args, **kwargs)
        except StopIteration:
            if 'logger' in kwargs:
                kwargs['logger']().error("Stop iteration triggered in real time mode. This is not expected.")
            raise StopIteration
        except Exception:
            if 'logger' in kwargs:
                kwargs['logger']().error("Unexpected error running clock tick.", exc_info=True)
            raise
        finally:
            return returned_value

    return call


def try_except(function: Callable):
    """
    Decorator wrapping an async function with exception handling

    :param Callable function: Period in nanosecond
    """
    def call(*args, **kwargs):
        returned_value = None
        try:
            returned_value = function(*args, **kwargs)
        except StopIteration:
            if 'logger' in kwargs:
                kwargs['logger']().error("Stop iteration triggered in real time mode. This is not expected.")
            raise StopIteration
        except Exception:
            if 'logger' in kwargs:
                kwargs['logger']().error("Unexpected error running clock tick.", exc_info=True)
            raise
        finally:
            return returned_value
    return call


@try_except
def in_executor(function: Callable):
    """
    Decorator wrapping a function with a call within an asyncio executor

    :param Callable function: Period in nanosecond
    """
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        loop = asyncio.get_event_loop()
        func = functools.partial(function, *args, **kwargs)
        return loop.run_in_executor(executor=None, func=func)

    return wrapped
