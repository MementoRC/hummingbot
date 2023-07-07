import asyncio
import unittest
from logging import Logger, LogRecord
from test.logger_mxin import LogLevel, TestLoggerMixin

from hummingbot.logger import HummingbotLogger


class TestTestLoggerMixin(unittest.TestCase):
    def setUp(self):
        self.logger = TestLoggerMixin()

    def test_handle(self):
        self.logger.log_records = []
        record = LogRecord(name="test", level=LogLevel.INFO, pathname="", lineno=0, msg="test message", args=None,
                           exc_info=None)
        self.logger.handle(record)
        self.assertEqual(len(self.logger.log_records), 1)
        self.assertEqual(self.logger.log_records[0].getMessage(), "test message")

    def test_is_logged(self):
        self.logger.log_records = []
        record = LogRecord(name="test", level=LogLevel.INFO, pathname="", lineno=0, msg="test message", args=None,
                           exc_info=None)
        self.logger.handle(record)
        self.assertTrue(self.logger.is_logged(LogLevel.INFO, "test message", ))
        self.assertFalse(self.logger.is_logged(LogLevel.ERROR, "test message", ))
        self.assertFalse(self.logger.is_logged(LogLevel.INFO, "other message", ))

        self.assertTrue(self.logger.is_logged("INFO", "test message", ))
        self.assertFalse(self.logger.is_logged("ERROR", "test message", ))
        self.assertFalse(self.logger.is_logged("INFO", "other message", ))

    def test_is_partially_logged(self):
        self.logger.log_records = []
        record = LogRecord(name="test", level=LogLevel.INFO, pathname="", lineno=0, msg="test message", args=None,
                           exc_info=None)
        self.logger.handle(record)
        self.assertTrue(self.logger.is_partially_logged(LogLevel.INFO, "test"))
        self.assertFalse(self.logger.is_partially_logged(LogLevel.ERROR, "test"))
        self.assertFalse(self.logger.is_partially_logged(LogLevel.INFO, "other"))

    def test_set_loggers_single_logger(self):
        logger = HummingbotLogger("TestLogger")
        logger.level = LogLevel.INFO
        self.assertNotEqual(1, logger.level)
        self.assertEqual(0, len(logger.handlers))

        self.logger.set_loggers([logger])
        self.assertEqual(1, logger.level)
        self.assertEqual(1, len(logger.handlers))
        self.assertEqual(self.logger, logger.handlers[0])

    def test_set_loggers_multiple_logger(self):
        loggers = []
        for i in range(5):
            logger = HummingbotLogger(f"TestLogger{i}")
            logger.level = LogLevel.INFO
            loggers.append(logger)
            self.assertNotEqual(1, logger.level)
            self.assertEqual(0, len(logger.handlers))

        self.logger.set_loggers(loggers)

        for logger in loggers:
            self.assertEqual(1, logger.level)
            self.assertEqual(1, len(logger.handlers))
            self.assertEqual(self.logger, logger.handlers[0])

    def test_set_loggers_other_logger(self):
        logger = Logger("TestLogger")
        logger.level = LogLevel.INFO
        self.assertNotEqual(logger.level, self.logger.level)
        self.assertEqual(0, len(logger.handlers))

        self.logger.set_loggers([logger])
        self.assertEqual(1, logger.level)
        self.assertEqual(1, len(logger.handlers))
        self.assertEqual(self.logger, logger.handlers[0])

    def test_set_loggers_some_none(self):
        loggers = [HummingbotLogger("Test"), None]
        self.logger.set_loggers(loggers)

    async def test_wait_for_logged(self):
        self.logger.log_records = []
        record = LogRecord(name="test", level=LogLevel.INFO, pathname="", lineno=0, msg="test message", args=None,
                           exc_info=None)
        self.logger.handle(record)

        # Test a message that has been logged
        logged = await self.logger.wait_for_logged(LogLevel.INFO, "test message", False, wait_s=0.1)
        self.assertIsNone(logged)

        # Test a message that has not been logged
        with self.assertRaises(asyncio.TimeoutError):
            await self.logger.wait_for_logged(LogLevel.INFO, "other message", False, wait_s=0.1)

    async def test_wait_for_logged_partial(self):
        self.logger.log_records = []
        record = LogRecord(name="test", level=LogLevel.INFO, pathname="", lineno=0, msg="test message", args=None,
                           exc_info=None)
        self.logger.handle(record)

        # Test a message that has been logged
        logged = await self.logger.wait_for_logged(LogLevel.INFO, "test", True, wait_s=0.1)
        self.assertIsNone(logged)

        # Test a message that has not been logged
        with self.assertRaises(asyncio.TimeoutError):
            await self.logger.wait_for_logged(LogLevel.INFO, "other", True, wait_s=0.1)


if __name__ == "__main__":
    unittest.main()
