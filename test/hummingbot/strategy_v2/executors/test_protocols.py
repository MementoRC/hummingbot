from unittest import TestCase

from hummingbot.strategy_v2.executors.protocols import (
    ExecutorConfigProtocol,
    ExecutorProtocol,
    ExecutorUpdateProtocol,
)


class _Config:
    def __init__(self):
        self.id = "cfg-1"
        self.type = "dummy"
        self.controller_id = None
        self.connector_name = "binance"
        self.trading_pair = "ETH-USDT"


class _Executor:
    def __init__(self):
        self.config = _Config()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _Update:
    def validate(self) -> bool:
        return True


class TestExecutorProtocols(TestCase):
    def test_config_protocol_accepts_conforming_object(self):
        self.assertIsInstance(_Config(), ExecutorConfigProtocol)

    def test_config_protocol_rejects_object_missing_attribute(self):
        config = _Config()
        del config.trading_pair
        self.assertNotIsInstance(config, ExecutorConfigProtocol)
        self.assertNotIsInstance(object(), ExecutorConfigProtocol)

    def test_executor_protocol_accepts_conforming_object(self):
        self.assertIsInstance(_Executor(), ExecutorProtocol)

    def test_executor_protocol_requires_config_and_lifecycle_methods(self):
        class NoStop:
            config = _Config()

            def start(self) -> None:
                pass

        class NoConfig:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        self.assertNotIsInstance(NoStop(), ExecutorProtocol)
        self.assertNotIsInstance(NoConfig(), ExecutorProtocol)

    def test_update_protocol(self):
        self.assertIsInstance(_Update(), ExecutorUpdateProtocol)
        self.assertNotIsInstance(object(), ExecutorUpdateProtocol)

    def test_update_protocol_is_method_only_so_supports_issubclass(self):
        self.assertTrue(issubclass(_Update, ExecutorUpdateProtocol))
        self.assertFalse(issubclass(_Config, ExecutorUpdateProtocol))
