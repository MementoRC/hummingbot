import asyncio
import concurrent.futures
import logging
import threading
import time

from hummingbot.logger import HummingbotLogger

lsb_logger = None


class AsyncioEventLoopThread(threading.Thread):

    @classmethod
    def logger(cls) -> HummingbotLogger:
        global lsb_logger
        if lsb_logger is None:
            lsb_logger = logging.getLogger(__name__)
        return lsb_logger

    def __init__(self, *args, loop=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hb_loop = asyncio.get_event_loop()
        self.loop = loop or asyncio.new_event_loop()
        self.running = False

    def run(self):
        self.running = True
        self.loop.run_forever()

    def run_coro(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, loop=self.loop)

    def run_and_wait_coro(self, coro, timeout=0.8):
        start_time = time.time()
        try:
            coro_result = self.run_coro(coro).result(timeout=timeout)
            return coro_result
        except concurrent.futures.TimeoutError:
            passed = time.time() - start_time
            self.logger().info(f"Error: scheduled {coro} did not return after {passed}")
            raise TimeoutError
        except concurrent.futures.CancelledError:
            self.logger().info(f"Error: scheduled {coro} was canceled")

    def run_list_coro(self, list_coro):
        return [asyncio.run_coroutine_threadsafe(coro, loop=self.loop) for coro in list_coro]

    def run_and_wait_for_list_coro(self, list_coro):
        futures = self.run_list_coro(list_coro)
        for i, task in enumerate(futures):
            try:
                print(f'Task {i} result: {task.result()}')
            except concurrent.futures.CancelledError:
                print(f'Task {i} cancelled')
        return futures

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.join()
        self.running = False
