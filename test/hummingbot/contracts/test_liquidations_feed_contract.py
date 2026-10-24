"""Contract: the pinned hb-liquidations-feed ``hb_compat`` feed satisfies ``LiquidationsFeedPort``.

The port is derived from ``controllers/generic/examples/liquidations_monitor_controller.py``,
which starts and stops the feed synchronously. At sub-package pin 4135fe56 the factory returns the
async-only core ``LiquidationsFeed`` (``start``/``stop`` are coroutines); the synchronous
``LiquidationsBase`` bridge only exists from hb-liquidations-feed deb3863 (#51) onward. The conformance test
feature-detects which variant is present and asserts the correct behaviour for each, so it holds
at any pin.
"""

import inspect

import pytest

from hummingbot.core.ports.liquidations import LiquidationsFeedPort, get_liquidations_feed
from test.hummingbot.contracts import assert_conforms

_COLUMNS = ["timestamp", "trading_pair", "quantity", "price", "side"]


@pytest.fixture
def feed() -> LiquidationsFeedPort:
    return get_liquidations_feed(connector="binance", trading_pairs=["BTC-USDT"], max_retention_seconds=10)


def test_feed_conforms_to_port(feed: LiquidationsFeedPort) -> None:
    lifecycle_is_async = any(inspect.iscoroutinefunction(getattr(feed, name)) for name in ("start", "stop"))
    if lifecycle_is_async:
        # Pre-deb3863 variant: the core async feed is returned, so the sync port must be rejected.
        with pytest.raises(AssertionError, match="must be sync"):
            assert_conforms(feed, LiquidationsFeedPort)
    else:
        assert_conforms(feed, LiquidationsFeedPort)


def test_feed_exposes_port_data_members(feed: LiquidationsFeedPort) -> None:
    assert feed.ready is False
    assert callable(feed.liquidations_df)
    assert callable(feed.start)
    assert callable(feed.stop)


def test_empty_liquidations_df_has_dataclass_columns(feed: LiquidationsFeedPort) -> None:
    assert list(feed.liquidations_df().columns) == _COLUMNS
    assert list(feed.liquidations_df("BTC-USDT").columns) == _COLUMNS
    assert feed.liquidations_df("UNKNOWN-PAIR").empty
