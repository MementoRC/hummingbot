"""Pure-Python port of composite_order_book.pyx.

CompositeOrderBook records orders filled during back-testing so it can
simulate order book consumption without mutating the real order book.
The bid_entries / ask_entries iterators yield composite entries where
the traded amount is subtracted from the original book quantity.
"""

from __future__ import annotations

from typing import Iterator

from hummingbot.core.data_type.common import TradeType
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.order_book_row import OrderBookRow


class CompositeOrderBook(OrderBook):
    """OrderBook variant that layers a 'traded' book on top of the real book."""

    def __init__(self, order_book: OrderBook = None) -> None:
        super().__init__()
        self._traded_order_book: OrderBook = OrderBook()

    @property
    def traded_order_book(self) -> OrderBook:
        return self._traded_order_book

    def clear_traded_order_book(self) -> None:
        self._traded_order_book._bid_book.clear()
        self._traded_order_book._bid_prices.clear()
        self._traded_order_book._ask_book.clear()
        self._traded_order_book._ask_prices.clear()

    def record_filled_order(self, order_fill_event: object) -> None:
        price: float = float(order_fill_event.price)
        amount: float = float(order_fill_event.amount)
        timestamp: int = int(order_fill_event.timestamp)

        traded = self._traded_order_book
        if order_fill_event.trade_type is TradeType.BUY:
            existing = traded._ask_book.get(price)
            if existing is not None:
                amount += existing[0]
            traded._apply_diffs([], [(price, amount, timestamp)], timestamp)
        elif order_fill_event.trade_type is TradeType.SELL:
            existing = traded._bid_book.get(price)
            if existing is not None:
                amount += existing[0]
            traded._apply_diffs([(price, amount, timestamp)], [], timestamp)

    def original_bid_entries(self) -> Iterator[OrderBookRow]:
        return super().bid_entries()

    def original_ask_entries(self) -> Iterator[OrderBookRow]:
        return super().ask_entries()

    def bid_entries(self) -> Iterator[OrderBookRow]:
        traded_bid = self._traded_order_book._bid_book
        cleanup_bids: list = []
        cleanup_asks: list = []

        for price in reversed(self._bid_prices):
            amount, uid = self._bid_book[price]
            traded = traded_bid.get(price)
            if traded is not None:
                traded_amount, traded_uid = traded
                composite = amount - traded_amount
                if composite > 0:
                    yield OrderBookRow(price, composite, uid)
                else:
                    # Traded amount exhausted this level — remove from traded book
                    cleanup_bids.append((price, min(amount, traded_amount), traded_uid))
            else:
                yield OrderBookRow(price, amount, uid)

        if cleanup_bids:
            self._traded_order_book._apply_diffs(cleanup_bids, cleanup_asks, self._last_diff_uid)

    def ask_entries(self) -> Iterator[OrderBookRow]:
        traded_ask = self._traded_order_book._ask_book
        cleanup_bids: list = []
        cleanup_asks: list = []

        for price in self._ask_prices:
            amount, uid = self._ask_book[price]
            traded = traded_ask.get(price)
            if traded is not None:
                traded_amount, traded_uid = traded
                composite = amount - traded_amount
                if composite > 0:
                    yield OrderBookRow(price, composite, uid)
                else:
                    cleanup_asks.append((price, min(amount, traded_amount), traded_uid))
            else:
                yield OrderBookRow(price, amount, uid)

        if cleanup_asks:
            self._traded_order_book._apply_diffs(cleanup_bids, cleanup_asks, self._last_diff_uid)

    def get_price(self, is_buy: bool) -> float:
        try:
            if is_buy:
                return next(iter(self.ask_entries())).price
            else:
                return next(iter(self.bid_entries())).price
        except StopIteration:
            raise EnvironmentError("Order book is empty - no price quote is possible.")
