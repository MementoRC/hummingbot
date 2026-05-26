"""Pure-Python LimitOrder. Drop-in API match for hummingbot.core.data_type.limit_order.LimitOrder."""

import time
from decimal import Decimal

import pandas as pd

from hummingbot.core.data_type.common import OrderType, PositionAction
from hummingbot.core.event.events import LimitOrderStatus


class LimitOrder:
    def __init__(
        self,
        client_order_id: str,
        trading_pair: str,
        is_buy: bool,
        base_currency: str,
        quote_currency: str,
        price: Decimal,
        quantity: Decimal,
        filled_quantity: Decimal = Decimal("NaN"),
        creation_timestamp: int = 0,
        status: LimitOrderStatus = LimitOrderStatus.UNKNOWN,
        position: PositionAction = PositionAction.NIL,
    ) -> None:
        self._client_order_id = client_order_id
        self._trading_pair = trading_pair
        self._is_buy = is_buy
        self._base_currency = base_currency
        self._quote_currency = quote_currency
        self._price = price
        self._quantity = quantity
        self._filled_quantity = filled_quantity
        self._creation_timestamp = creation_timestamp
        self._status = status
        self._position = position

    @property
    def client_order_id(self) -> str:
        return self._client_order_id

    @property
    def trading_pair(self) -> str:
        return self._trading_pair

    @property
    def is_buy(self) -> bool:
        return self._is_buy

    @property
    def base_currency(self) -> str:
        return self._base_currency

    @property
    def quote_currency(self) -> str:
        return self._quote_currency

    @property
    def price(self) -> Decimal:
        return self._price

    @property
    def quantity(self) -> Decimal:
        return self._quantity

    @property
    def filled_quantity(self) -> Decimal:
        return self._filled_quantity

    @property
    def creation_timestamp(self) -> int:
        return self._creation_timestamp

    @property
    def status(self) -> LimitOrderStatus:
        return self._status

    @property
    def position(self) -> PositionAction:
        return self._position

    def age_til(self, end_timestamp: int) -> int:
        start_timestamp = 0
        if self._creation_timestamp > 0:
            start_timestamp = self._creation_timestamp
        elif len(self._client_order_id) > 16 and self._client_order_id[-16:].isnumeric():
            start_timestamp = int(self._client_order_id[-16:])
        if 0 < start_timestamp < end_timestamp:
            return int((end_timestamp - start_timestamp) / 1e6)
        return -1

    def age(self) -> int:
        return self.age_til(int(time.time() * 1e6))

    def order_type(self) -> OrderType:
        return OrderType.LIMIT

    def copy_with_id(self, client_order_id: str) -> "LimitOrder":
        return LimitOrder(
            client_order_id=client_order_id,
            trading_pair=self._trading_pair,
            is_buy=self._is_buy,
            base_currency=self._base_currency,
            quote_currency=self._quote_currency,
            price=self._price,
            quantity=self._quantity,
            filled_quantity=self._filled_quantity,
            creation_timestamp=self._creation_timestamp,
            status=self._status,
            position=self._position,
        )

    def __repr__(self) -> str:
        return (
            f"LimitOrder('{self._client_order_id}', '{self._trading_pair}', {self._is_buy}, "
            f"'{self._base_currency}', '{self._quote_currency}', {self._price}, {self._quantity}, "
            f"{self._filled_quantity}, {self._creation_timestamp})"
        )

    def __lt__(self, other: "LimitOrder") -> bool:
        if self._price == other._price:
            return self._client_order_id < other._client_order_id
        return self._price < other._price

    @classmethod
    def to_pandas(
        cls,
        limit_orders: list["LimitOrder"],
        mid_price: float = 0.0,
        hanging_ids: list[str] | None = None,
        end_time_order_age: int = 0,
    ) -> pd.DataFrame:
        buys = [o for o in limit_orders if o.is_buy]
        sells = [o for o in limit_orders if not o.is_buy]
        buys.sort(key=lambda x: x.price, reverse=True)
        sells.sort(key=lambda x: x.price, reverse=True)
        columns = ["Order ID", "Type", "Price", "Spread", "Amount", "Age", "Hang"]
        data = []
        now_timestamp = int(time.time() * 1e6) if end_time_order_age == 0 else end_time_order_age
        sells.extend(buys)
        for order in sells:
            order_id_txt = (
                order.client_order_id if len(order.client_order_id) <= 7 else f"...{order.client_order_id[-4:]}"
            )
            type_txt = "buy" if order.is_buy else "sell"
            price = float(order.price)
            spread_txt = f"{(0 if mid_price == 0 else abs(float(order.price) - mid_price) / mid_price):.2%}"
            quantity = float(order.quantity)
            age_txt = "n/a"
            age_seconds = order.age_til(now_timestamp)
            if age_seconds >= 0:
                age_txt = pd.Timestamp(age_seconds, unit="s", tz="UTC").strftime("%H:%M:%S")
            hang_txt = "n/a" if hanging_ids is None else ("yes" if order.client_order_id in hanging_ids else "no")
            data.append([order_id_txt, type_txt, price, spread_txt, quantity, age_txt, hang_txt])
        return pd.DataFrame(data=data, columns=columns)
