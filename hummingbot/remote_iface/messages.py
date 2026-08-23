from typing import Any

from pydantic import BaseModel


class MQTT_STATUS_CODE:
    ERROR: int = 400
    SUCCESS: int = 200


class PubSubMessage(BaseModel):
    """Base class for pub/sub messages.

    Local replacement for ``commlib.msg.PubSubMessage`` (a bare pydantic
    ``BaseModel``). Kept so the wire format and field semantics are identical
    after dropping the commlib dependency.
    """

    pass


class RPCMessage(BaseModel):
    """Namespace base for RPC request/response messages.

    Local replacement for ``commlib.msg.RPCMessage``: a ``BaseModel`` exposing
    nested ``Request``/``Response`` ``BaseModel`` classes for subclasses to
    extend.
    """

    class Request(BaseModel):
        pass

    class Response(BaseModel):
        pass


class NotifyMessage(PubSubMessage):
    seq: int | None = 0
    timestamp: int | None = -1
    msg: str | None = ""


class StatusUpdateMessage(PubSubMessage):
    timestamp: int | None = -1
    type: str | None = ""
    msg: str | None = ""


class InternalEventMessage(PubSubMessage):
    timestamp: int | None = -1
    type: str | None = "ievent"
    data: dict | None = {}


class LogMessage(PubSubMessage):
    timestamp: float = 0.0
    msg: str = ""
    level_no: int = 0
    level_name: str = ""
    logger_name: str = ""


class ExternalEventMessage(PubSubMessage):
    timestamp: int | None = -1
    sequence: int | None = 0
    type: str | None = "eevent"
    data: dict[str, Any] | None = {}


class StartCommandMessage(RPCMessage):
    class Request(RPCMessage.Request):
        log_level: str | None = None
        script: str | None = None
        conf: str | None = None
        is_quickstart: bool | None = False
        async_backend: bool | None = True

    class Response(RPCMessage.Response):
        status: int | None = MQTT_STATUS_CODE.SUCCESS
        msg: str | None = ""


class StopCommandMessage(RPCMessage):
    class Request(RPCMessage.Request):
        skip_order_cancellation: bool | None = False
        async_backend: bool | None = True

    class Response(RPCMessage.Response):
        status: int | None = MQTT_STATUS_CODE.SUCCESS
        msg: str | None = ""


class ConfigCommandMessage(RPCMessage):
    class Request(RPCMessage.Request):
        params: list[tuple[str, Any]] | None = []

    class Response(RPCMessage.Response):
        changes: list[tuple[str, Any]] | None = []
        config: dict[str, Any] | None = {}
        status: int | None = MQTT_STATUS_CODE.SUCCESS
        msg: str | None = ""


class ImportCommandMessage(RPCMessage):
    class Request(RPCMessage.Request):
        strategy: str

    class Response(RPCMessage.Response):
        status: int | None = MQTT_STATUS_CODE.SUCCESS
        msg: str | None = ""


class StatusCommandMessage(RPCMessage):
    class Request(RPCMessage.Request):
        async_backend: bool | None = True

    class Response(RPCMessage.Response):
        status: int | None = MQTT_STATUS_CODE.SUCCESS
        msg: str | None = ""
        data: Any | None = ""


class HistoryCommandMessage(RPCMessage):
    class Request(RPCMessage.Request):
        days: float | None = 0
        verbose: bool | None = False
        precision: int | None = None
        async_backend: bool | None = True

    class Response(RPCMessage.Response):
        status: int | None = MQTT_STATUS_CODE.SUCCESS
        msg: str | None = ""
        trades: list[Any] | None = []


class BalanceLimitCommandMessage(RPCMessage):
    class Request(RPCMessage.Request):
        exchange: str
        asset: str
        amount: float

    class Response(RPCMessage.Response):
        status: int | None = MQTT_STATUS_CODE.SUCCESS
        msg: str | None = ""
        data: str | None = ""


class BalancePaperCommandMessage(RPCMessage):
    class Request(RPCMessage.Request):
        asset: str
        amount: float

    class Response(RPCMessage.Response):
        status: int | None = MQTT_STATUS_CODE.SUCCESS
        msg: str | None = ""
        data: str | None = ""
