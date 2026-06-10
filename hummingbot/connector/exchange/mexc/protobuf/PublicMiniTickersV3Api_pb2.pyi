from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

from google.protobuf import descriptor as _descriptor, message as _message
from google.protobuf.internal import containers as _containers

from hummingbot.connector.exchange.mexc.protobuf import PublicMiniTickerV3Api_pb2 as _PublicMiniTickerV3Api_pb2

DESCRIPTOR: _descriptor.FileDescriptor

class PublicMiniTickersV3Api(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[_PublicMiniTickerV3Api_pb2.PublicMiniTickerV3Api]
    def __init__(
        self, items: _Optional[_Iterable[_Union[_PublicMiniTickerV3Api_pb2.PublicMiniTickerV3Api, _Mapping]]] = ...
    ) -> None: ...
