from typing import TYPE_CHECKING, Dict, Optional

from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.order_book_row import OrderBookRow

if TYPE_CHECKING:
    from data_type_primitives import OrderBookSnapshot


def apply_snapshot_to_book(book: OrderBook, snapshot: "OrderBookSnapshot") -> None:
    """Replace the contents of ``book`` with ``snapshot``, using the snapshot timestamp (ms) as update id."""
    update_id = snapshot.timestamp
    bids = [OrderBookRow(float(e.price), float(e.quantity), update_id) for e in snapshot.bids]
    asks = [OrderBookRow(float(e.price), float(e.quantity), update_id) for e in snapshot.asks]
    book.apply_snapshot(bids, asks, update_id)


class OrderBookCache:
    """One parent OrderBook per pair; a snapshot is re-applied only when its timestamp changed."""

    def __init__(self) -> None:
        self._books: Dict[str, OrderBook] = {}
        self._stamps: Dict[str, int] = {}

    def sync(self, trading_pair: str, snapshot: Optional["OrderBookSnapshot"]) -> OrderBook:
        """Return the cached book for the pair (empty when there is no snapshot yet), refreshed from snapshot."""
        book = self._books.get(trading_pair)
        if book is None:
            book = OrderBook()
            self._books[trading_pair] = book
        if snapshot is not None and self._stamps.get(trading_pair) != snapshot.timestamp:
            apply_snapshot_to_book(book, snapshot)
            self._stamps[trading_pair] = snapshot.timestamp
        return book

    @property
    def books(self) -> Dict[str, OrderBook]:
        return dict(self._books)

    def clear(self) -> None:
        self._books.clear()
        self._stamps.clear()
