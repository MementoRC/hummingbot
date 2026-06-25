from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Generic, NamedTuple, TypeVar

from pydantic_core import core_schema


class OrderType(Enum):
    MARKET = 1
    LIMIT = 2
    LIMIT_MAKER = 3
    AMM_SWAP = 4
    AMM_ADD = 5  # Add liquidity to AMM/CLMM pool
    AMM_REMOVE = 6  # Remove liquidity from AMM/CLMM pool
    # Conditional order types (exchange-native)
    STOP_LOSS = 7
    TAKE_PROFIT = 8
    TRAILING_STOP = 9
    STOP_LOSS_LIMIT = 10
    TAKE_PROFIT_LIMIT = 11
    TRAILING_STOP_LIMIT = 12

    def is_limit_type(self):
        return self in (
            OrderType.LIMIT,
            OrderType.LIMIT_MAKER,
            OrderType.STOP_LOSS_LIMIT,
            OrderType.TAKE_PROFIT_LIMIT,
            OrderType.TRAILING_STOP_LIMIT,
        )

    def is_delayed_market_type(self):
        """Returns True for conditional orders that trigger a market execution."""
        return self in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT, OrderType.TRAILING_STOP)

    def is_conditional_type(self):
        """Returns True for any conditional/triggered order type."""
        return self in (
            OrderType.STOP_LOSS,
            OrderType.TAKE_PROFIT,
            OrderType.TRAILING_STOP,
            OrderType.STOP_LOSS_LIMIT,
            OrderType.TAKE_PROFIT_LIMIT,
            OrderType.TRAILING_STOP_LIMIT,
        )


class OpenOrder(NamedTuple):
    client_order_id: str
    trading_pair: str
    price: Decimal
    amount: Decimal
    executed_amount: Decimal
    status: str
    order_type: OrderType
    is_buy: bool
    time: int
    exchange_order_id: str


class PositionAction(Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    NIL = "NIL"


# For Derivatives Exchanges
class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


# For Derivatives Exchanges
class PositionMode(Enum):
    HEDGE = "HEDGE"
    ONEWAY = "ONEWAY"


class PriceType(Enum):
    MidPrice = 1
    BestBid = 2
    BestAsk = 3
    LastTrade = 4
    LastOwnTrade = 5
    InventoryCost = 6
    Custom = 7


class TradeType(Enum):
    BUY = 1
    SELL = 2
    RANGE = 3


class LPType(Enum):
    ADD = 1
    REMOVE = 2
    COLLECT = 3


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class GroupedSetDict(dict[_KT, set[_VT]]):
    def add_or_update(self, key: _KT, *args: _VT) -> "GroupedSetDict":
        if key in self:
            self[key].update(args)
        else:
            self[key] = set(args)
        return self

    def remove(self, key: _KT, value: _VT) -> "GroupedSetDict":
        if key in self:
            self[key].discard(value)
            if not self[key]:  # If set becomes empty, remove the key
                del self[key]
        return self

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Any,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls, core_schema.dict_schema(core_schema.any_schema(), core_schema.set_schema(core_schema.any_schema()))
        )


MarketDict = GroupedSetDict[str, set[str]]


# TODO? : Allow pulling the hash for _KT via a lambda so that things like type can be a key?
class LazyDict(dict[_KT, _VT], Generic[_KT, _VT]):
    def __init__(self, default_value_factory: Callable[[_KT], _VT] = None):
        super().__init__()
        self.default_value_factory = default_value_factory

    def __missing__(self, key: _KT) -> _VT:
        if self.default_value_factory is None:
            raise KeyError(f"Key {key} not found in {self} and no default value factory is set")
        self[key] = self.default_value_factory(key)
        return self[key]

    def get(self, key: _KT) -> _VT:
        if key in self:
            return self[key]
        return self.__missing__(key)

    def get_or_add(self, key: _KT, value_factory: Callable[[], _VT]) -> _VT:
        if key not in self:
            self[key] = value_factory()
        return self[key]
