from unittest import TestCase

from pydantic import BaseModel, ValidationError

from hummingbot.remote_iface.messages import (
    MQTT_STATUS_CODE,
    BalanceLimitCommandMessage,
    BalancePaperCommandMessage,
    ConfigCommandMessage,
    ExternalEventMessage,
    HistoryCommandMessage,
    ImportCommandMessage,
    InternalEventMessage,
    LogMessage,
    NotifyMessage,
    PubSubMessage,
    RPCMessage,
    StartCommandMessage,
    StatusCommandMessage,
    StatusUpdateMessage,
    StopCommandMessage,
)

RPC_COMMANDS = [
    StartCommandMessage,
    StopCommandMessage,
    ConfigCommandMessage,
    ImportCommandMessage,
    StatusCommandMessage,
    HistoryCommandMessage,
    BalanceLimitCommandMessage,
    BalancePaperCommandMessage,
]

PUBSUB_MESSAGES = [
    NotifyMessage,
    StatusUpdateMessage,
    InternalEventMessage,
    LogMessage,
    ExternalEventMessage,
]


class MQTTStatusCodeTests(TestCase):
    def test_values(self):
        self.assertEqual(400, MQTT_STATUS_CODE.ERROR)
        self.assertEqual(200, MQTT_STATUS_CODE.SUCCESS)


class BaseClassTests(TestCase):
    def test_pubsub_base(self):
        self.assertTrue(issubclass(PubSubMessage, BaseModel))
        msg = PubSubMessage()
        self.assertEqual({}, msg.model_dump())
        self.assertEqual(msg, PubSubMessage.model_validate(msg.model_dump()))

    def test_rpc_base(self):
        self.assertTrue(issubclass(RPCMessage, BaseModel))
        self.assertTrue(issubclass(RPCMessage.Request, BaseModel))
        self.assertTrue(issubclass(RPCMessage.Response, BaseModel))
        self.assertEqual({}, RPCMessage().model_dump())
        for nested in (RPCMessage.Request, RPCMessage.Response):
            obj = nested()
            self.assertEqual({}, obj.model_dump())
            self.assertEqual(obj, nested.model_validate(obj.model_dump()))

    def test_inventory_counts(self):
        self.assertEqual(8, len(RPC_COMMANDS))
        self.assertEqual(5, len(PUBSUB_MESSAGES))
        for cls in RPC_COMMANDS:
            self.assertTrue(issubclass(cls, RPCMessage))
            self.assertTrue(issubclass(cls.Request, RPCMessage.Request))
            self.assertTrue(issubclass(cls.Response, RPCMessage.Response))
        for cls in PUBSUB_MESSAGES:
            self.assertTrue(issubclass(cls, PubSubMessage))


class PubSubMessageTests(TestCase):
    def assert_round_trip(self, obj):
        self.assertEqual(obj, type(obj).model_validate(obj.model_dump()))

    def test_notify_message(self):
        self.assertEqual({"seq": 0, "timestamp": -1, "msg": ""}, NotifyMessage().model_dump())
        obj = NotifyMessage(seq=3, timestamp=1700000000, msg="hello")
        self.assertEqual({"seq": 3, "timestamp": 1700000000, "msg": "hello"}, obj.model_dump())
        self.assert_round_trip(NotifyMessage())
        self.assert_round_trip(obj)

    def test_status_update_message(self):
        self.assertEqual({"timestamp": -1, "type": "", "msg": ""}, StatusUpdateMessage().model_dump())
        obj = StatusUpdateMessage(timestamp=5, type="info", msg="ok")
        self.assertEqual({"timestamp": 5, "type": "info", "msg": "ok"}, obj.model_dump())
        self.assert_round_trip(StatusUpdateMessage())
        self.assert_round_trip(obj)

    def test_internal_event_message(self):
        self.assertEqual({"timestamp": -1, "type": "ievent", "data": {}}, InternalEventMessage().model_dump())
        obj = InternalEventMessage(timestamp=7, type="custom", data={"a": 1})
        self.assertEqual({"timestamp": 7, "type": "custom", "data": {"a": 1}}, obj.model_dump())
        self.assert_round_trip(InternalEventMessage())
        self.assert_round_trip(obj)

    def test_internal_event_message_default_data_not_shared(self):
        first = InternalEventMessage()
        first.data["x"] = 1
        self.assertEqual({}, InternalEventMessage().data)

    def test_log_message(self):
        self.assertEqual(
            {"timestamp": 0.0, "msg": "", "level_no": 0, "level_name": "", "logger_name": ""},
            LogMessage().model_dump(),
        )
        obj = LogMessage(timestamp=1.5, msg="boom", level_no=40, level_name="ERROR", logger_name="hb")
        self.assertEqual(
            {"timestamp": 1.5, "msg": "boom", "level_no": 40, "level_name": "ERROR", "logger_name": "hb"},
            obj.model_dump(),
        )
        self.assert_round_trip(LogMessage())
        self.assert_round_trip(obj)

    def test_external_event_message(self):
        self.assertEqual(
            {"timestamp": -1, "sequence": 0, "type": "eevent", "data": {}},
            ExternalEventMessage().model_dump(),
        )
        obj = ExternalEventMessage(timestamp=9, sequence=2, type="ext", data={"k": [1, 2], "n": None})
        self.assertEqual(
            {"timestamp": 9, "sequence": 2, "type": "ext", "data": {"k": [1, 2], "n": None}},
            obj.model_dump(),
        )
        self.assert_round_trip(ExternalEventMessage())
        self.assert_round_trip(obj)


class RPCCommandMessageTests(TestCase):
    def assert_round_trip(self, obj):
        self.assertEqual(obj, type(obj).model_validate(obj.model_dump()))

    def test_command_wrapper_has_no_fields(self):
        for cls in RPC_COMMANDS:
            with self.subTest(cls=cls.__name__):
                self.assertEqual({}, cls().model_dump())

    def test_start(self):
        req = StartCommandMessage.Request()
        self.assertEqual(
            {"log_level": None, "script": None, "conf": None, "is_quickstart": False, "async_backend": True},
            req.model_dump(),
        )
        explicit = StartCommandMessage.Request(
            log_level="DEBUG", script="s.py", conf="c.yml", is_quickstart=True, async_backend=False
        )
        self.assertEqual(
            {"log_level": "DEBUG", "script": "s.py", "conf": "c.yml", "is_quickstart": True, "async_backend": False},
            explicit.model_dump(),
        )
        self.assert_round_trip(req)
        self.assert_round_trip(explicit)
        self.assertEqual({"status": 200, "msg": ""}, StartCommandMessage.Response().model_dump())
        resp = StartCommandMessage.Response(status=MQTT_STATUS_CODE.ERROR, msg="bad")
        self.assertEqual({"status": 400, "msg": "bad"}, resp.model_dump())
        self.assert_round_trip(StartCommandMessage.Response())
        self.assert_round_trip(resp)

    def test_stop(self):
        req = StopCommandMessage.Request()
        self.assertEqual({"skip_order_cancellation": False, "async_backend": True}, req.model_dump())
        explicit = StopCommandMessage.Request(skip_order_cancellation=True, async_backend=False)
        self.assertEqual({"skip_order_cancellation": True, "async_backend": False}, explicit.model_dump())
        self.assert_round_trip(req)
        self.assert_round_trip(explicit)
        self.assertEqual({"status": 200, "msg": ""}, StopCommandMessage.Response().model_dump())
        resp = StopCommandMessage.Response(status=400, msg="fail")
        self.assertEqual({"status": 400, "msg": "fail"}, resp.model_dump())
        self.assert_round_trip(StopCommandMessage.Response())
        self.assert_round_trip(resp)

    def test_config(self):
        req = ConfigCommandMessage.Request()
        self.assertEqual({"params": []}, req.model_dump())
        explicit = ConfigCommandMessage.Request(params=[("a", 1), ("b", "x")])
        self.assertEqual({"params": [("a", 1), ("b", "x")]}, explicit.model_dump())
        self.assert_round_trip(req)
        self.assert_round_trip(explicit)
        self.assertEqual(
            {"changes": [], "config": {}, "status": 200, "msg": ""},
            ConfigCommandMessage.Response().model_dump(),
        )
        resp = ConfigCommandMessage.Response(
            changes=[("a", 2)], config={"a": 2}, status=MQTT_STATUS_CODE.ERROR, msg="m"
        )
        self.assertEqual(
            {"changes": [("a", 2)], "config": {"a": 2}, "status": 400, "msg": "m"},
            resp.model_dump(),
        )
        self.assert_round_trip(ConfigCommandMessage.Response())
        self.assert_round_trip(resp)

    def test_import(self):
        with self.assertRaises(ValidationError):
            ImportCommandMessage.Request()
        req = ImportCommandMessage.Request(strategy="pmm")
        self.assertEqual({"strategy": "pmm"}, req.model_dump())
        self.assert_round_trip(req)
        self.assertEqual({"status": 200, "msg": ""}, ImportCommandMessage.Response().model_dump())
        resp = ImportCommandMessage.Response(status=400, msg="nope")
        self.assertEqual({"status": 400, "msg": "nope"}, resp.model_dump())
        self.assert_round_trip(ImportCommandMessage.Response())
        self.assert_round_trip(resp)

    def test_status(self):
        req = StatusCommandMessage.Request()
        self.assertEqual({"async_backend": True}, req.model_dump())
        explicit = StatusCommandMessage.Request(async_backend=False)
        self.assertEqual({"async_backend": False}, explicit.model_dump())
        self.assert_round_trip(req)
        self.assert_round_trip(explicit)
        self.assertEqual({"status": 200, "msg": "", "data": ""}, StatusCommandMessage.Response().model_dump())
        resp = StatusCommandMessage.Response(status=400, msg="m", data={"k": [1]})
        self.assertEqual({"status": 400, "msg": "m", "data": {"k": [1]}}, resp.model_dump())
        self.assert_round_trip(StatusCommandMessage.Response())
        self.assert_round_trip(resp)

    def test_history(self):
        req = HistoryCommandMessage.Request()
        self.assertEqual({"days": 0, "verbose": False, "precision": None, "async_backend": True}, req.model_dump())
        explicit = HistoryCommandMessage.Request(days=1.5, verbose=True, precision=4, async_backend=False)
        self.assertEqual(
            {"days": 1.5, "verbose": True, "precision": 4, "async_backend": False},
            explicit.model_dump(),
        )
        self.assert_round_trip(req)
        self.assert_round_trip(explicit)
        self.assertEqual({"status": 200, "msg": "", "trades": []}, HistoryCommandMessage.Response().model_dump())
        resp = HistoryCommandMessage.Response(status=400, msg="m", trades=[{"id": 1}, "t"])
        self.assertEqual({"status": 400, "msg": "m", "trades": [{"id": 1}, "t"]}, resp.model_dump())
        self.assert_round_trip(HistoryCommandMessage.Response())
        self.assert_round_trip(resp)

    def test_balance_limit(self):
        with self.assertRaises(ValidationError):
            BalanceLimitCommandMessage.Request()
        req = BalanceLimitCommandMessage.Request(exchange="binance", asset="BTC", amount=1.25)
        self.assertEqual({"exchange": "binance", "asset": "BTC", "amount": 1.25}, req.model_dump())
        self.assert_round_trip(req)
        self.assertEqual({"status": 200, "msg": "", "data": ""}, BalanceLimitCommandMessage.Response().model_dump())
        resp = BalanceLimitCommandMessage.Response(status=400, msg="m", data="d")
        self.assertEqual({"status": 400, "msg": "m", "data": "d"}, resp.model_dump())
        self.assert_round_trip(BalanceLimitCommandMessage.Response())
        self.assert_round_trip(resp)

    def test_balance_paper(self):
        with self.assertRaises(ValidationError):
            BalancePaperCommandMessage.Request(asset="BTC")
        req = BalancePaperCommandMessage.Request(asset="USDT", amount=100.0)
        self.assertEqual({"asset": "USDT", "amount": 100.0}, req.model_dump())
        self.assert_round_trip(req)
        self.assertEqual({"status": 200, "msg": "", "data": ""}, BalancePaperCommandMessage.Response().model_dump())
        resp = BalancePaperCommandMessage.Response(status=400, msg="m", data="d")
        self.assertEqual({"status": 400, "msg": "m", "data": "d"}, resp.model_dump())
        self.assert_round_trip(BalancePaperCommandMessage.Response())
        self.assert_round_trip(resp)
