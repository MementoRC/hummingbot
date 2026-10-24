"""Liquidations feed port.

Derived only from what ``controllers/generic/examples/liquidations_monitor_controller.py`` uses.
Backed by ``liquidations_feed.hb_compat`` (hb-liquidations-feed); the in-tree
``data_feed/liquidations_feed`` package is the legacy implementation.
"""

from collections.abc import Iterable
from typing import Protocol

from pandas import DataFrame


class LiquidationsFeedPort(Protocol):
    """Synchronous-lifecycle liquidations feed, as the legacy ``LiquidationsBase`` consumers use it."""

    @property
    def ready(self) -> bool: ...

    def liquidations_df(self, trading_pair: str | None = None) -> DataFrame: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


def get_liquidations_feed(
    connector: str, trading_pairs: Iterable[str], max_retention_seconds: int
) -> LiquidationsFeedPort:
    """Build a liquidations feed through the hb-liquidations-feed ``hb_compat`` factory."""
    from liquidations_feed.hb_compat import LiquidationsConfig, LiquidationsFactory

    config = LiquidationsConfig(
        connector=connector,
        trading_pairs=set(trading_pairs),
        max_retention_seconds=max_retention_seconds,
    )
    return LiquidationsFactory.get_liquidations_feed(config)
