"""Tests for hummingbot.core.data_type.order_expiration_entry_rust.

In CI (Rust not built), _ACCELERATED=False and OrderExpirationEntry is
the pure-Python fallback class. These tests cover the full fallback
implementation so diff-cover reports 100% on this module.
"""

import pytest

from hummingbot.core.data_type.order_expiration_entry_rust import OrderExpirationEntry, is_accelerated

# ---------------------------------------------------------------------------
# is_accelerated
# ---------------------------------------------------------------------------


def test_is_accelerated_returns_bool():
    assert isinstance(is_accelerated(), bool)


# ---------------------------------------------------------------------------
# Construction and properties
# ---------------------------------------------------------------------------


@pytest.fixture()
def entry():
    return OrderExpirationEntry("BTC-USDT", "order-1", 1_700_000_000.0, 1_700_003_600.0)


def test_trading_pair(entry):
    assert entry.trading_pair == "BTC-USDT"


def test_order_id(entry):
    assert entry.order_id == "order-1"


def test_timestamp(entry):
    assert entry.timestamp == pytest.approx(1_700_000_000.0)


def test_expiration_timestamp(entry):
    assert entry.expiration_timestamp == pytest.approx(1_700_003_600.0)


def test_timestamp_coerced_to_float():
    e = OrderExpirationEntry("ETH-USDT", "o2", 1000, 2000)
    assert isinstance(e.timestamp, float)
    assert isinstance(e.expiration_timestamp, float)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_contains_key_fields(entry):
    r = repr(entry)
    assert "OrderExpirationEntry" in r
    assert "BTC-USDT" in r
    assert "order-1" in r


# ---------------------------------------------------------------------------
# __lt__ — ordering semantics
# ---------------------------------------------------------------------------


def test_lt_by_expiration_timestamp():
    earlier = OrderExpirationEntry("BTC-USDT", "o1", 1.0, 100.0)
    later = OrderExpirationEntry("BTC-USDT", "o2", 1.0, 200.0)
    assert earlier < later
    assert not later < earlier


def test_lt_tie_break_by_order_id():
    a = OrderExpirationEntry("BTC-USDT", "aaa", 1.0, 100.0)
    b = OrderExpirationEntry("BTC-USDT", "bbb", 1.0, 100.0)
    assert a < b
    assert not b < a


def test_lt_equal_entries_not_less_than():
    e1 = OrderExpirationEntry("BTC-USDT", "o1", 1.0, 100.0)
    e2 = OrderExpirationEntry("BTC-USDT", "o1", 1.0, 100.0)
    assert not e1 < e2
    assert not e2 < e1


def test_sort_uses_lt():
    entries = [
        OrderExpirationEntry("X", "z", 0.0, 300.0),
        OrderExpirationEntry("X", "a", 0.0, 100.0),
        OrderExpirationEntry("X", "m", 0.0, 200.0),
    ]
    sorted_entries = sorted(entries)
    assert sorted_entries[0].expiration_timestamp == 100.0
    assert sorted_entries[1].expiration_timestamp == 200.0
    assert sorted_entries[2].expiration_timestamp == 300.0


# ---------------------------------------------------------------------------
# __eq__ and __hash__
# ---------------------------------------------------------------------------


def test_eq_identical_entries():
    e1 = OrderExpirationEntry("BTC-USDT", "o1", 1.0, 100.0)
    e2 = OrderExpirationEntry("BTC-USDT", "o1", 1.0, 100.0)
    assert e1 == e2


def test_eq_different_trading_pair():
    e1 = OrderExpirationEntry("BTC-USDT", "o1", 1.0, 100.0)
    e2 = OrderExpirationEntry("ETH-USDT", "o1", 1.0, 100.0)
    assert e1 != e2


def test_eq_non_entry_returns_not_implemented():
    e = OrderExpirationEntry("X", "o1", 1.0, 100.0)
    result = e.__eq__("not an entry")
    assert result is NotImplemented


def test_hash_equal_entries_same_hash():
    e1 = OrderExpirationEntry("BTC-USDT", "o1", 1.0, 100.0)
    e2 = OrderExpirationEntry("BTC-USDT", "o1", 1.0, 100.0)
    assert hash(e1) == hash(e2)


def test_hash_usable_in_set():
    e1 = OrderExpirationEntry("BTC-USDT", "o1", 1.0, 100.0)
    e2 = OrderExpirationEntry("BTC-USDT", "o1", 1.0, 100.0)
    s = {e1, e2}
    assert len(s) == 1


# ---------------------------------------------------------------------------
# to_pandas classmethod
# ---------------------------------------------------------------------------


def test_to_pandas_columns():
    entries = [OrderExpirationEntry("BTC-USDT", "o1", 1.0, 100.0)]
    df = OrderExpirationEntry.to_pandas(entries)
    assert list(df.columns) == ["trading_pair", "order_id", "timestamp", "expiration_timestamp"]


def test_to_pandas_single_entry(entry):
    df = OrderExpirationEntry.to_pandas([entry])
    assert len(df) == 1
    assert df.iloc[0]["trading_pair"] == "BTC-USDT"
    assert df.iloc[0]["order_id"] == "order-1"
    assert df.iloc[0]["timestamp"] == pytest.approx(1_700_000_000.0)
    assert df.iloc[0]["expiration_timestamp"] == pytest.approx(1_700_003_600.0)


def test_to_pandas_multiple_entries():
    entries = [
        OrderExpirationEntry("BTC-USDT", "o1", 1.0, 100.0),
        OrderExpirationEntry("ETH-USDT", "o2", 2.0, 200.0),
    ]
    df = OrderExpirationEntry.to_pandas(entries)
    assert len(df) == 2
    assert df.iloc[1]["trading_pair"] == "ETH-USDT"


def test_to_pandas_empty_list():
    df = OrderExpirationEntry.to_pandas([])
    assert len(df) == 0
    assert list(df.columns) == ["trading_pair", "order_id", "timestamp", "expiration_timestamp"]
