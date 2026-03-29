"""Auto-skip hummingbot candles tests superseded by the candles-feed sub-package.

When the candles_feed package is installed (e.g., via pixi workspace), exchange
adapter tests that have equivalent coverage in the sub-package are skipped to
avoid redundant CI time. Tests for exchanges NOT covered by the sub-package
(aevo, bitget, bitmart, btc_markets, dexalot, pacifica) continue to run.

To force all tests to run (e.g., for upstream compatibility checks):
    pytest --run-superseded
"""

import pytest

# Exchanges with adapters in the candles-feed sub-package
_SUBPACKAGE_EXCHANGES = {
    "ascend_ex",
    "binance",
    "bybit",
    "gate_io",
    "hyperliquid",
    "kraken",
    "kucoin",
    "mexc",
    "okx",
}

# Files that test shared infrastructure superseded by sub-package compat tests
_SUPERSEDED_INFRA_FILES = {
    "test_candles_base.py",
    "test_candles_factory.py",
}


def _candles_feed_available():
    try:
        import candles_feed  # noqa: F401

        return True
    except ImportError:
        return False


def pytest_addoption(parser):
    parser.addoption(
        "--run-superseded",
        action="store_true",
        default=False,
        help="Run candles tests even when superseded by candles-feed sub-package",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-superseded"):
        return
    if not _candles_feed_available():
        return

    skip_marker = pytest.mark.skip(reason="Superseded by candles-feed sub-package (use --run-superseded to force)")

    for item in items:
        # Skip infrastructure tests (test_candles_base, test_candles_factory)
        filename = item.path.name if hasattr(item, "path") else item.fspath.basename
        if filename in _SUPERSEDED_INFRA_FILES:
            item.add_marker(skip_marker)
            continue

        # Skip exchange tests where sub-package has equivalent adapter
        for exchange in _SUBPACKAGE_EXCHANGES:
            if f"{exchange}_" in str(item.path if hasattr(item, "path") else item.fspath):
                item.add_marker(skip_marker)
                break
