"""L3 (modular-tier) conformance check: hb-market-data's LiveMarketData satisfies
hb-strategy-framework's MarketDataProtocol. Moved here from hb-market-data PR #34
because only the parent installs both packages together. Imports are hard on purpose.
"""

import inspect
import unittest
from unittest.mock import MagicMock

from market_data import LiveMarketData
from strategy_framework.protocols.market_data import MarketDataProtocol

PROTOCOL_METHODS = ("get_mid_price", "get_order_book_snapshot", "get_candles")


def _params(func):
    return [(p.name, p.kind, p.default) for p in inspect.signature(func).parameters.values()]


def _annotation_names(func):
    return {
        key: value
        if isinstance(value, str)
        else (getattr(value, "__name__", None) or repr(value)).removeprefix("typing.")
        for key, value in inspect.get_annotations(func).items()
    }


class TestMarketDataProtocolConformance(unittest.TestCase):
    def test_isinstance_with_runtime_checkable_protocol(self):
        self.assertIsInstance(LiveMarketData(MagicMock(), "binance"), MarketDataProtocol)

    def test_method_signatures_match_protocol(self):
        for name in PROTOCOL_METHODS:
            with self.subTest(method=name):
                proto_fn = getattr(MarketDataProtocol, name)
                impl_fn = getattr(LiveMarketData, name)
                self.assertEqual(_params(proto_fn), _params(impl_fn))
                self.assertEqual(inspect.iscoroutinefunction(proto_fn), inspect.iscoroutinefunction(impl_fn))

    def test_annotations_match_protocol(self):
        # Class identity is not checked here: annotation types are TYPE_CHECKING-only imports, so they cannot be resolved at runtime.
        for name in PROTOCOL_METHODS:
            with self.subTest(method=name):
                self.assertEqual(
                    _annotation_names(getattr(MarketDataProtocol, name)),
                    _annotation_names(getattr(LiveMarketData, name)),
                )
