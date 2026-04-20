"""Root conftest.py — exclude test modules with missing upstream dependencies."""

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
