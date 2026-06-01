import bisect
import logging
import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from hummingbot.core.data_type.order_book_message import OrderBookMessage
from hummingbot.core.data_type.order_book_query_result import OrderBookQueryResult
from hummingbot.core.data_type.order_book_row import OrderBookRow
from hummingbot.core.event.events import OrderBookEvent, OrderBookTradeEvent
from hummingbot.core.pubsub import PubSub
from hummingbot.logger import HummingbotLogger

ob_logger: Optional[HummingbotLogger] = None
NaN = float("nan")


def _truncate_overlap_dex(
    bid_book: Dict[float, Tuple[float, int]],
    ask_book: Dict[float, Tuple[float, int]],
    bid_prices: list,
    ask_prices: list,
) -> None:
    """Dex variant: larger notional value wins."""
    while bid_prices and ask_prices:
        top_bid_price = bid_prices[-1]
        top_ask_price = ask_prices[0]
        if top_bid_price < top_ask_price:
            break
        bid_amount, _ = bid_book[top_bid_price]
        ask_amount, _ = ask_book[top_ask_price]
        if bid_amount * top_bid_price > ask_amount * top_ask_price:
            del ask_book[top_ask_price]
            ask_prices.pop(0)
        else:
            del bid_book[top_bid_price]
            bid_prices.pop()


def _truncate_overlap_centralised(
    bid_book: Dict[float, Tuple[float, int]],
    ask_book: Dict[float, Tuple[float, int]],
    bid_prices: list,
    ask_prices: list,
) -> None:
    """Centralised variant: newer update_id wins."""
    while bid_prices and ask_prices:
        top_bid_price = bid_prices[-1]
        top_ask_price = ask_prices[0]
        if top_bid_price < top_ask_price:
            break
        _, bid_uid = bid_book[top_bid_price]
        _, ask_uid = ask_book[top_ask_price]
        if bid_uid > ask_uid:
            del ask_book[top_ask_price]
            ask_prices.pop(0)
        else:
            del bid_book[top_bid_price]
            bid_prices.pop()


class OrderBook(PubSub):
    ORDER_BOOK_TRADE_EVENT_TAG = OrderBookEvent.TradeEvent.value

    @classmethod
    def logger(cls) -> HummingbotLogger:
        global ob_logger
        if ob_logger is None:
            ob_logger = logging.getLogger(__name__)
        return ob_logger

    def __init__(self, dex: bool = False) -> None:
        super().__init__()
        # Bid book: price -> (amount, update_id); bids sorted ascending by price
        self._bid_book: Dict[float, Tuple[float, int]] = {}
        self._bid_prices: list = []  # sorted ascending; best bid = [-1]
        # Ask book: price -> (amount, update_id); asks sorted ascending by price
        self._ask_book: Dict[float, Tuple[float, int]] = {}
        self._ask_prices: list = []  # sorted ascending; best ask = [0]
        self._snapshot_uid: int = 0
        self._last_diff_uid: int = 0
        self._best_bid: float = NaN
        self._best_ask: float = NaN
        self._last_trade_price: float = NaN
        self._last_applied_trade: float = -1000.0
        self._last_trade_price_rest_updated: float = -1000.0
        self._dex: bool = dex

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _book_insert(
        self,
        book: Dict[float, Tuple[float, int]],
        prices: list,
        price: float,
        amount: float,
        update_id: int,
    ) -> None:
        if price not in book:
            bisect.insort(prices, price)
        book[price] = (amount, update_id)

    def _book_remove(
        self,
        book: Dict[float, Tuple[float, int]],
        prices: list,
        price: float,
    ) -> None:
        if price in book:
            del book[price]
            idx = bisect.bisect_left(prices, price)
            if idx < len(prices) and prices[idx] == price:
                prices.pop(idx)

    # ------------------------------------------------------------------
    # Core cdef → private Python methods
    # ------------------------------------------------------------------

    def _apply_diffs(
        self,
        bids: List[Tuple[float, float, int]],
        asks: List[Tuple[float, float, int]],
        update_id: int,
    ) -> None:
        bid_book = self._bid_book
        bid_prices = self._bid_prices
        ask_book = self._ask_book
        ask_prices = self._ask_prices

        for price, amount, uid in bids:
            if price in bid_book:
                self._book_remove(bid_book, bid_prices, price)
            if amount > 0:
                self._book_insert(bid_book, bid_prices, price, amount, uid)

        for price, amount, uid in asks:
            if price in ask_book:
                self._book_remove(ask_book, ask_prices, price)
            if amount > 0:
                self._book_insert(ask_book, ask_prices, price, amount, uid)

        if self._dex:
            _truncate_overlap_dex(bid_book, ask_book, bid_prices, ask_prices)
        else:
            _truncate_overlap_centralised(bid_book, ask_book, bid_prices, ask_prices)

        self._best_bid = bid_prices[-1] if bid_prices else NaN
        self._best_ask = ask_prices[0] if ask_prices else NaN
        self._last_diff_uid = update_id

    def _apply_snapshot(
        self,
        bids: List[Tuple[float, float, int]],
        asks: List[Tuple[float, float, int]],
        update_id: int,
    ) -> None:
        self._bid_book.clear()
        self._bid_prices.clear()
        self._ask_book.clear()
        self._ask_prices.clear()

        bid_book = self._bid_book
        bid_prices = self._bid_prices
        ask_book = self._ask_book
        ask_prices = self._ask_prices

        for price, amount, uid in bids:
            self._book_insert(bid_book, bid_prices, price, amount, uid)
        for price, amount, uid in asks:
            self._book_insert(ask_book, ask_prices, price, amount, uid)

        if self._dex:
            _truncate_overlap_dex(bid_book, ask_book, bid_prices, ask_prices)
            self._best_bid = bid_prices[-1] if bid_prices else NaN
            self._best_ask = ask_prices[0] if ask_prices else NaN
        else:
            self._best_bid = bid_prices[-1] if bid_prices else NaN
            self._best_ask = ask_prices[0] if ask_prices else NaN

        self._snapshot_uid = update_id

    def _apply_trade(self, trade_event: object) -> None:
        self._last_trade_price = trade_event.price
        self._last_applied_trade = time.perf_counter()
        self.c_trigger_event(self.ORDER_BOOK_TRADE_EVENT_TAG, trade_event)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def last_trade_price(self) -> float:
        return self._last_trade_price

    @last_trade_price.setter
    def last_trade_price(self, value: float) -> None:
        self._last_trade_price = value

    @property
    def last_applied_trade(self) -> float:
        return self._last_applied_trade

    @property
    def last_trade_price_rest_updated(self) -> float:
        return self._last_trade_price_rest_updated

    @last_trade_price_rest_updated.setter
    def last_trade_price_rest_updated(self, value: float) -> None:
        self._last_trade_price_rest_updated = value

    @property
    def snapshot_uid(self) -> int:
        return self._snapshot_uid

    @property
    def last_diff_uid(self) -> int:
        return self._last_diff_uid

    @property
    def snapshot(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        bids_rows = list(self.bid_entries())
        asks_rows = list(self.ask_entries())
        bids_df = pd.DataFrame(data=bids_rows, columns=OrderBookRow._fields, dtype="float64")
        asks_df = pd.DataFrame(data=asks_rows, columns=OrderBookRow._fields, dtype="float64")
        return bids_df, asks_df

    # ------------------------------------------------------------------
    # Public apply_* API
    # ------------------------------------------------------------------

    def apply_diffs(self, bids: List[OrderBookRow], asks: List[OrderBookRow], update_id: int) -> None:
        self._apply_diffs(
            [(r.price, r.amount, r.update_id) for r in bids],
            [(r.price, r.amount, r.update_id) for r in asks],
            update_id,
        )

    def apply_snapshot(self, bids: List[OrderBookRow], asks: List[OrderBookRow], update_id: int) -> None:
        self._apply_snapshot(
            [(r.price, r.amount, r.update_id) for r in bids],
            [(r.price, r.amount, r.update_id) for r in asks],
            update_id,
        )

    def apply_trade(self, trade: OrderBookTradeEvent) -> None:
        self._apply_trade(trade)

    def apply_pandas_diffs(self, bids_df: pd.DataFrame, asks_df: pd.DataFrame) -> None:
        """Diffs data frame must have 3 columns [price, amount, update_id] with float64 dtype."""
        self.apply_numpy_diffs(bids_df.values, asks_df.values)

    def apply_numpy_diffs(self, bids_array: np.ndarray, asks_array: np.ndarray) -> None:
        """Diffs array must have 3 columns [price, amount, update_id] with float64 dtype."""
        last_update_id = 0
        bids: List[Tuple[float, float, int]] = []
        for row in bids_array:
            uid = int(row[2])
            bids.append((row[0], row[1], uid))
            if uid > last_update_id:
                last_update_id = uid
        asks: List[Tuple[float, float, int]] = []
        for row in asks_array:
            uid = int(row[2])
            asks.append((row[0], row[1], uid))
            if uid > last_update_id:
                last_update_id = uid
        self._apply_diffs(bids, asks, last_update_id)

    def apply_numpy_snapshot(self, bids_array: np.ndarray, asks_array: np.ndarray) -> None:
        """Snapshot array must have 3 columns [price, amount, update_id] with float64 dtype."""
        last_update_id = 0
        bids: List[Tuple[float, float, int]] = []
        for row in bids_array:
            uid = int(row[2])
            bids.append((row[0], row[1], uid))
            if uid > last_update_id:
                last_update_id = uid
        asks: List[Tuple[float, float, int]] = []
        for row in asks_array:
            uid = int(row[2])
            asks.append((row[0], row[1], uid))
            if uid > last_update_id:
                last_update_id = uid
        self._apply_snapshot(bids, asks, last_update_id)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def bid_entries(self) -> Iterator[OrderBookRow]:
        bid_book = self._bid_book
        for price in reversed(self._bid_prices):
            amount, uid = bid_book[price]
            yield OrderBookRow(price, amount, uid)

    def ask_entries(self) -> Iterator[OrderBookRow]:
        ask_book = self._ask_book
        for price in self._ask_prices:
            amount, uid = ask_book[price]
            yield OrderBookRow(price, amount, uid)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_buy(self, amount: float) -> List[OrderBookRow]:
        amount_left = amount
        retval = []
        for ask_entry in self.ask_entries():
            if ask_entry.amount < amount_left:
                retval.append(ask_entry)
                amount_left -= ask_entry.amount
            else:
                retval.append(OrderBookRow(ask_entry.price, amount_left, ask_entry.update_id))
                amount_left = 0.0
                break
        return retval

    def simulate_sell(self, amount: float) -> List[OrderBookRow]:
        amount_left = amount
        retval = []
        for bid_entry in self.bid_entries():
            if bid_entry.amount < amount_left:
                retval.append(bid_entry)
                amount_left -= bid_entry.amount
            else:
                retval.append(OrderBookRow(bid_entry.price, amount_left, bid_entry.update_id))
                amount_left = 0.0
                break
        return retval

    # ------------------------------------------------------------------
    # Price queries
    # ------------------------------------------------------------------

    def get_price(self, is_buy: bool) -> float:
        if is_buy:
            if not self._ask_prices:
                raise EnvironmentError("Order book is empty - no price quote is possible.")
            return self._best_ask
        else:
            if not self._bid_prices:
                raise EnvironmentError("Order book is empty - no price quote is possible.")
            return self._best_bid

    def get_price_for_volume(self, is_buy: bool, volume: float) -> OrderBookQueryResult:
        cumulative_volume = 0.0
        result_price = NaN
        entries = self.ask_entries() if is_buy else self.bid_entries()
        for row in entries:
            cumulative_volume += row.amount
            if cumulative_volume >= volume:
                result_price = row.price
                break
        return OrderBookQueryResult(NaN, volume, result_price, min(cumulative_volume, volume))

    def get_vwap_for_volume(self, is_buy: bool, volume: float) -> OrderBookQueryResult:
        total_cost = 0.0
        total_volume = 0.0
        result_vwap = NaN
        entries = self.ask_entries() if is_buy else self.bid_entries()
        for row in entries:
            total_cost += row.amount * row.price
            total_volume += row.amount
            if total_volume >= volume:
                total_cost -= row.amount * row.price
                total_volume -= row.amount
                incremental = volume - total_volume
                total_cost += incremental * row.price
                total_volume += incremental
                result_vwap = total_cost / total_volume
                break
        return OrderBookQueryResult(NaN, volume, result_vwap, min(total_volume, volume))

    def get_price_for_quote_volume(self, is_buy: bool, quote_volume: float) -> OrderBookQueryResult:
        cumulative_volume = 0.0
        result_price = NaN
        entries = self.ask_entries() if is_buy else self.bid_entries()
        for row in entries:
            cumulative_volume += row.amount * row.price
            if cumulative_volume >= quote_volume:
                result_price = row.price
                break
        return OrderBookQueryResult(NaN, quote_volume, result_price, min(cumulative_volume, quote_volume))

    def get_quote_volume_for_base_amount(self, is_buy: bool, base_amount: float) -> OrderBookQueryResult:
        cumulative_volume = 0.0
        cumulative_base = 0.0
        entries = self.ask_entries() if is_buy else self.bid_entries()
        for row in entries:
            row_amount = row.amount
            if row_amount + cumulative_base >= base_amount:
                row_amount = base_amount - cumulative_base
            cumulative_base += row_amount
            cumulative_volume += row_amount * row.price
            if cumulative_base >= base_amount:
                break
        return OrderBookQueryResult(NaN, base_amount, NaN, cumulative_volume)

    def get_volume_for_price(self, is_buy: bool, price: float) -> OrderBookQueryResult:
        cumulative_volume = 0.0
        result_price = NaN
        if is_buy:
            for row in self.ask_entries():
                if row.price > price:
                    break
                cumulative_volume += row.amount
                result_price = row.price
        else:
            for row in self.bid_entries():
                if row.price < price:
                    break
                cumulative_volume += row.amount
                result_price = row.price
        return OrderBookQueryResult(price, NaN, result_price, cumulative_volume)

    def get_quote_volume_for_price(self, is_buy: bool, price: float) -> OrderBookQueryResult:
        cumulative_volume = 0.0
        result_price = NaN
        if is_buy:
            for row in self.ask_entries():
                if row.price > price:
                    break
                cumulative_volume += row.amount * row.price
                result_price = row.price
        else:
            for row in self.bid_entries():
                if row.price < price:
                    break
                cumulative_volume += row.amount * row.price
                result_price = row.price
        return OrderBookQueryResult(price, NaN, result_price, cumulative_volume)

    # ------------------------------------------------------------------
    # c_* aliases — required for callers in exchange_base.py that call
    # order_book.c_XXX() directly (C-level in Cython, Python aliases here)
    # ------------------------------------------------------------------

    def c_get_price(self, is_buy: bool) -> float:
        return self.get_price(is_buy)

    def c_get_price_for_volume(self, is_buy: bool, volume: float) -> OrderBookQueryResult:
        return self.get_price_for_volume(is_buy, volume)

    def c_get_vwap_for_volume(self, is_buy: bool, volume: float) -> OrderBookQueryResult:
        return self.get_vwap_for_volume(is_buy, volume)

    def c_get_price_for_quote_volume(self, is_buy: bool, quote_volume: float) -> OrderBookQueryResult:
        return self.get_price_for_quote_volume(is_buy, quote_volume)

    def c_get_quote_volume_for_base_amount(self, is_buy: bool, base_amount: float) -> OrderBookQueryResult:
        return self.get_quote_volume_for_base_amount(is_buy, base_amount)

    def c_get_volume_for_price(self, is_buy: bool, price: float) -> OrderBookQueryResult:
        return self.get_volume_for_price(is_buy, price)

    def c_get_quote_volume_for_price(self, is_buy: bool, price: float) -> OrderBookQueryResult:
        return self.get_quote_volume_for_price(is_buy, price)

    # ------------------------------------------------------------------
    # Restore from snapshot + diffs
    # ------------------------------------------------------------------

    def restore_from_snapshot_and_diffs(self, snapshot: OrderBookMessage, diffs: List[OrderBookMessage]) -> None:
        replay_position = bisect.bisect_right(diffs, snapshot)
        replay_diffs = diffs[replay_position:]
        self.apply_snapshot(snapshot.bids, snapshot.asks, snapshot.update_id)
        for diff in replay_diffs:
            self.apply_diffs(diff.bids, diff.asks, diff.update_id)
