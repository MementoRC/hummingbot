"""Root conftest.py — exclude test modules with missing upstream dependencies."""

import os as _os

# Remove stale .so files for modules converted from Cython to pure Python.
# The .so has import priority over .py; delete it so the .py is used directly.
_CONVERTED_SO_FILES = [
    # clock: converted from Cython in C19; .so compiled with c_start/c_stop/c_tick
    # which no longer exist after C20.
    "hummingbot/core/clock.cpython-312-x86_64-linux-gnu.so",
    "hummingbot/strategy/order_tracker.cpython-312-x86_64-linux-gnu.so",
    # strategy_base / strategy_py_base: converted from Cython; .so compiled
    # against old TimeIterator C-struct (104 bytes) which is now pure Python (88 bytes).
    "hummingbot/strategy/strategy_base.cpython-312-x86_64-linux-gnu.so",
    "hummingbot/strategy/strategy_py_base.cpython-312-x86_64-linux-gnu.so",
    # trading_intensity: converted from Cython; .so compiled against old
    # EventListener C-struct (48 bytes) which is now pure Python (16 bytes).
    "hummingbot/strategy/__utils__/trailing_indicators/trading_intensity.cpython-312-x86_64-linux-gnu.so",
    # api_asset_price_delegate / order_book_asset_price_delegate: .pyx removed.
    "hummingbot/strategy/api_asset_price_delegate.cpython-312-x86_64-linux-gnu.so",
    "hummingbot/strategy/order_book_asset_price_delegate.cpython-312-x86_64-linux-gnu.so",
    # cross_exchange_market_making order_id_market_pair_tracker: .pyx removed.
    "hummingbot/strategy/cross_exchange_market_making/order_id_market_pair_tracker.cpython-312-x86_64-linux-gnu.so",
    # transaction_tracker: .pyx removed.
    "hummingbot/core/data_type/transaction_tracker.cpython-312-x86_64-linux-gnu.so",
]
for _so in _CONVERTED_SO_FILES:
    if _os.path.exists(_so):
        _os.remove(_so)


def _can_import(modname: str) -> bool:
    try:
        __import__(modname)
        return True
    except ImportError:
        return False


# Directories/files to skip during collection unconditionally (missing upstream deps)
_SKIP_PATHS = (
    "connector/derivative/decibel_perpetual",
    "connector/gateway/test_gateway_lp.py",
    # Out-of-scope strategies removed in Phase C (source .pyx deleted)
    "strategy/pure_market_making",
    "strategy/avellaneda_market_making",
    "strategy/cross_exchange_mining",
)

# Directories to skip only when their optional dependency is not installed
_CONDITIONAL_SKIPS: list[str] = []
if not _can_import("lighter"):
    _CONDITIONAL_SKIPS.append("connector/derivative/lighter_perpetual")
if not _can_import("v4_proto"):
    _CONDITIONAL_SKIPS.append("connector/derivative/dydx_v4_perpetual")
if not _can_import("eip712_structs"):
    _CONDITIONAL_SKIPS.append("connector/exchange/vertex")


def pytest_ignore_collect(collection_path, config):
    """Skip test paths with missing upstream dependencies."""
    path_str = str(collection_path)
    return any(skip in path_str for skip in _SKIP_PATHS) or any(skip in path_str for skip in _CONDITIONAL_SKIPS)
