"""Canonical trading data-type primitives.

These definitions live in the ``data-type-primitives`` sub-package, an L0 leaf
under ADR 0001, and are re-exported here so the historical import path
``hummingbot.core.data_type.common`` keeps working unchanged.

They are re-exported rather than redefined because ``Enum`` equality is
identity-based: a structurally identical second definition compares unequal to
the first. Two copies of ``TradeType``/``PositionAction`` would therefore make
``TradeFeeBase`` select the wrong fee class silently, with no import error.
Exactly one class object must exist per type.
"""

from data_type_primitives.common import (
    GroupedSetDict,
    LazyDict,
    LPType,
    MarketDict,
    OpenOrder,
    OrderType,
    PositionAction,
    PositionMode,
    PositionSide,
    PriceType,
    TradeType,
)

__all__ = [
    "GroupedSetDict",
    "LPType",
    "LazyDict",
    "MarketDict",
    "OpenOrder",
    "OrderType",
    "PositionAction",
    "PositionMode",
    "PositionSide",
    "PriceType",
    "TradeType",
]
