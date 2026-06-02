"""Root conftest.py — exclude test modules with missing upstream dependencies."""

import os as _os

# Remove stale .so files for modules converted from Cython to pure Python.
# The .so has import priority over .py; delete it so the .py is used directly.
_CONVERTED_SO_FILES = [
    "hummingbot/strategy/order_tracker.cpython-312-x86_64-linux-gnu.so",
]
for _so in _CONVERTED_SO_FILES:
    if _os.path.exists(_so):
        _os.remove(_so)

# Directories/files to skip during collection (missing upstream deps)
_SKIP_PATHS = (
    "connector/derivative/decibel_perpetual",
    "connector/derivative/dydx_v4_perpetual",
    "connector/exchange/vertex",
    "connector/gateway/test_gateway_lp.py",
)


def pytest_ignore_collect(collection_path, config):
    """Skip test paths with missing upstream dependencies."""
    path_str = str(collection_path)
    return any(skip in path_str for skip in _SKIP_PATHS)
