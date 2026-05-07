import asyncio
import threading
import time
import unittest

from hummingbot.core.utils.async_utils import call_sync


class CallSyncCrossThreadTests(unittest.TestCase):
    """Regression tests for call_sync cross-thread routing (Case A).

    These tests guard against the deadlock that occurred when commlib's
    paho-mqtt callback thread invoked call_sync while the test main thread
    was driving the same loop via run_until_complete — the old implementation
    fell through to loop.run_until_complete from the worker thread, which
    hangs on a running loop.
    """

    def setUp(self):
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_started = threading.Event()
        self._loop_thread.start()
        self._loop_started.wait(timeout=5.0)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon_threadsafe(self._loop_started.set)
        self._loop.run_forever()

    def tearDown(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5.0)
        self._loop.close()

    # ------------------------------------------------------------------
    # Case A: loop is running on worker thread; we call from main thread
    # ------------------------------------------------------------------

    def test_call_sync_from_non_loop_thread_returns_result(self):
        """call_sync routes via run_coroutine_threadsafe and returns the coro result."""

        async def coro():
            return 42

        start = time.monotonic()
        result = call_sync(coro(), loop=self._loop, timeout=5.0)
        elapsed = time.monotonic() - start

        self.assertEqual(result, 42)
        self.assertLess(elapsed, 1.0, "call_sync should complete in well under 1s")

    def test_call_sync_from_non_loop_thread_propagates_exception(self):
        """call_sync propagates exceptions raised inside the coroutine."""

        async def failing_coro():
            raise ValueError("boom")

        with self.assertRaises(ValueError, msg="Expected ValueError from coro"):
            call_sync(failing_coro(), loop=self._loop, timeout=5.0)

    # ------------------------------------------------------------------
    # Case C: illegal recursive call from inside the running loop
    # ------------------------------------------------------------------

    def test_call_sync_from_inside_loop_raises_runtime_error(self):
        """call_sync raises RuntimeError when invoked from within the target loop."""

        async def inner():
            return 99

        async def outer():
            # This is the illegal scenario: call_sync from inside the loop itself.
            return call_sync(inner(), loop=self._loop, timeout=5.0)

        future = asyncio.run_coroutine_threadsafe(outer(), self._loop)
        with self.assertRaises(RuntimeError):
            future.result(timeout=5.0)


if __name__ == "__main__":
    unittest.main()
