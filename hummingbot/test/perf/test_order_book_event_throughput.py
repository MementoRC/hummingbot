"""R1 gate: measure OrderBook event dispatch throughput before composition refactor.

Captures a perf baseline for the legacy ``cdef class OrderBook(PubSub)`` so the
post-refactor pure-Python OrderBook (C6) can be compared apples-to-apples.

Plan deviation: the original plan pseudo-code called ``ob.c_trigger_event(0, ...)``
directly. ``c_trigger_event`` is declared ``cdef`` in pubsub.pyx (C-only, not
Python-callable). The Python entry point is ``def trigger_event(event_tag: Enum,
msg)`` which wraps the cdef call. We measure ``trigger_event`` because that is
the path Python callers actually traverse and is what C6 will need to match.
"""

import time
from enum import IntEnum

import pytest


class _BenchEvent(IntEnum):
    Dispatch = 0


@pytest.mark.benchmark
def test_legacy_orderbook_event_throughput():
    """Establish baseline: current cdef class OrderBook(PubSub) event throughput."""
    from hummingbot.core.data_type.order_book import OrderBook

    ob = OrderBook()
    iters = 100_000
    start = time.perf_counter()
    for i in range(iters):
        ob.trigger_event(_BenchEvent.Dispatch, {"price": i, "amount": 1.0})
    elapsed = time.perf_counter() - start
    rate = iters / elapsed
    print(f"\nLEGACY OrderBook event rate: {rate:,.0f} events/s (iters={iters}, elapsed={elapsed:.3f}s)")
    # No hard assertion — this captures baseline; comparison happens in C6.
