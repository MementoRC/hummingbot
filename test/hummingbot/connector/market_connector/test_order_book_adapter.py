from decimal import Decimal
import unittest

import pytest

pytest.importorskip("market_connector")

from data_type_primitives import OrderBookEntry, OrderBookSnapshot  # noqa: E402

from hummingbot.connector.market_connector.order_book_adapter import OrderBookCache  # noqa: E402


def _snapshot(timestamp: int, bid: str = "99", ask: str = "101") -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=timestamp,
        bids=[
            OrderBookEntry(price=Decimal(bid), quantity=Decimal("2")),
            OrderBookEntry(price=Decimal("98"), quantity=Decimal("1")),
        ],
        asks=[OrderBookEntry(price=Decimal(ask), quantity=Decimal("3"))],
    )


class OrderBookCacheTests(unittest.TestCase):
    def test_snapshot_is_converted(self):
        book = OrderBookCache().sync("ETH-USDC", _snapshot(1000))
        self.assertEqual(99.0, book.get_price(False))
        self.assertEqual(101.0, book.get_price(True))
        self.assertEqual([99.0, 98.0], [row.price for row in book.bid_entries()])
        self.assertEqual(1000, book.snapshot_uid)

    def test_missing_snapshot_gives_empty_book(self):
        book = OrderBookCache().sync("ETH-USDC", None)
        self.assertEqual([], list(book.bid_entries()))

    def test_same_book_instance_reused_and_updated_on_new_timestamp(self):
        cache = OrderBookCache()
        first = cache.sync("ETH-USDC", _snapshot(1000))
        second = cache.sync("ETH-USDC", _snapshot(2000, bid="99.5"))
        self.assertIs(first, second)
        self.assertEqual(99.5, second.get_price(False))

    def test_unchanged_timestamp_is_not_reapplied(self):
        cache = OrderBookCache()
        book = cache.sync("ETH-USDC", _snapshot(1000))
        cache.sync("ETH-USDC", _snapshot(1000, bid="50"))
        self.assertEqual(99.0, book.get_price(False))

    def test_clear_drops_books(self):
        cache = OrderBookCache()
        cache.sync("ETH-USDC", _snapshot(1000))
        cache.clear()
        self.assertEqual({}, cache.books)
