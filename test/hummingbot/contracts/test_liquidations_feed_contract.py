"""Contract: the pinned hb-liquidations-feed ``hb_compat`` feed satisfies ``LiquidationsFeedPort``.

The port is derived from ``controllers/generic/examples/liquidations_monitor_controller.py``,
which starts and stops the feed synchronously. At sub-package pin 4135fe56 the factory returns the
async-only core ``LiquidationsFeed`` (``start``/``stop`` are coroutines); the synchronous
``LiquidationsBase`` bridge only exists from hb-liquidations-feed deb3863 (#51) onward. That gap is
pinned by a strict xfail so that the pin bump flips it to XPASS and fails loudly until the marker is
removed.
"""

import pytest

from hummingbot.core.ports.liquidations import LiquidationsFeedPort, get_liquidations_feed
from test.hummingbot.contracts import assert_conforms

_COLUMNS = ["timestamp", "trading_pair", "quantity", "price", "side"]


@pytest.fixture
def feed() -> LiquidationsFeedPort:
    return get_liquidations_feed(connector="binance", trading_pairs=["BTC-USDT"], max_retention_seconds=10)


@pytest.mark.xfail(
    strict=True,
    reason="pin 4135fe56: start()/stop() are async; sync LiquidationsBase bridge lands in hb-liquidations-feed>=deb3863",
)
def test_feed_conforms_to_port(feed: LiquidationsFeedPort) -> None:
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
