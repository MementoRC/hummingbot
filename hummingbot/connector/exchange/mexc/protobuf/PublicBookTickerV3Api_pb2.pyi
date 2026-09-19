from typing import ClassVar as _ClassVar, Optional as _Optional

from google.protobuf import descriptor as _descriptor, message as _message

DESCRIPTOR: _descriptor.FileDescriptor

class PublicBookTickerV3Api(_message.Message):
    __slots__ = ("bidPrice", "bidQuantity", "askPrice", "askQuantity")
    BIDPRICE_FIELD_NUMBER: _ClassVar[int]
    BIDQUANTITY_FIELD_NUMBER: _ClassVar[int]
    ASKPRICE_FIELD_NUMBER: _ClassVar[int]
    ASKQUANTITY_FIELD_NUMBER: _ClassVar[int]
    bidPrice: str
    bidQuantity: str
    askPrice: str
    askQuantity: str
    def __init__(
        self,
        bidPrice: _Optional[str] = ...,
        bidQuantity: _Optional[str] = ...,
        askPrice: _Optional[str] = ...,
        askQuantity: _Optional[str] = ...,
    ) -> None: ...
