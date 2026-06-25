"""Root conftest: Phase-C event artifact cleanup + sub-package supersede skip.

PHASE-C CLEANUP  (runs at conftest import time, before any test collection)
---------------
EventListener (C1), EventLogger (C2), EventReporter (C3) were converted to
pure Python.  Any compiled .so built against the old Cython cdef-class
EventListener (48-byte struct) fails with "size changed, may indicate binary
incompatibility" at import time.  The pure-Python .py replacements exist for
every module listed here; removing the stale .so files lets Python fall
through to the .py versions for the test session.

AUTO-SKIP  (original functionality)
------------------------------------
Sub-packages declare which hummingbot test paths they replace via
[tool.hummingbot.supersedes] in their pyproject.toml:

    [tool.hummingbot.supersedes]
    test_paths = ["test/hummingbot/data_feed/candles_feed"]
    exchanges = ["binance", "bybit", ...]

When a sub-package is importable, tests under its declared test_paths
are skipped — but only for exchanges/modules listed in its `exchanges`
array. Tests for HB-only modules keep running unconditionally.

To force all tests to run:
    pytest --run-superseded

Adding a new sub-package requires NO changes here — just add the
[tool.hummingbot.supersedes] section to the sub-package's pyproject.toml.
"""

import importlib
from pathlib import Path
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# ---------------------------------------------------------------------------
# Phase-C: remove stale compiled artifacts for event subsystem conversions.
# Pure-Python .py replacements exist for every module listed here.
# ---------------------------------------------------------------------------
_HBOT = Path(__file__).parents[1] / "hummingbot"
_PHASE_C_STEMS = [
    # C1 — EventListener base class
    "core/event/event_listener",
    # C2 — EventLogger
    "core/event/event_logger",
    # C3 — EventReporter
    "core/event/event_reporter",
]
_PHASE_C_ORPHANED_SO = [
    _so for _stem in _PHASE_C_STEMS for _so in (_HBOT / _stem).parent.glob(f"{(_HBOT / _stem).name}.cpython-*.so")
]
for _so in _PHASE_C_ORPHANED_SO:
    _so.unlink()

# Evict any module that may have been cached before this conftest ran.
_EVICT = {
    "event_listener",
    "event_logger",
    "event_reporter",
}
for _key in list(sys.modules):
    if any(k in _key for k in _EVICT):
        del sys.modules[_key]


def _discover_superseded_tests():
    """Scan sub-packages/*/pyproject.toml for [tool.hummingbot.supersedes]."""
    repo_root = Path(__file__).parent.parent
    sub_packages_dir = repo_root / "sub-packages"

    if not sub_packages_dir.is_dir():
        return []

    results = []
    for pkg_dir in sub_packages_dir.iterdir():
        if not pkg_dir.is_dir():
            continue
        pyproject = pkg_dir / "pyproject.toml"
        if not pyproject.exists():
            continue

        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        supersedes = data.get("tool", {}).get("hummingbot", {}).get("supersedes", {})
        if not supersedes:
            continue

        # Determine the importable package name from project metadata
        project_name = data.get("project", {}).get("name", pkg_dir.name)
        # hb-candles-feed -> candles_feed
        import_name = project_name.removeprefix("hb-").replace("-", "_")

        # Check if the package is actually importable
        try:
            importlib.import_module(import_name)
        except ImportError:
            continue

        results.append(
            {
                "package": import_name,
                "test_paths": supersedes.get("test_paths", []),
                "exchanges": set(supersedes.get("exchanges", [])),
            }
        )

    return results


# Cache at module load time
_SUPERSEDED = _discover_superseded_tests()


def pytest_addoption(parser):
    parser.addoption(
        "--run-superseded",
        action="store_true",
        default=False,
        help="Run tests even when superseded by an installed sub-package",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-superseded"):
        return
    if not _SUPERSEDED:
        return

    for entry in _SUPERSEDED:
        pkg_name = entry["package"]
        test_paths = entry["test_paths"]
        exchanges = entry["exchanges"]

        skip_marker = pytest.mark.skip(reason=f"Superseded by {pkg_name} sub-package (use --run-superseded to force)")

        for item in items:
            item_path = str(item.path if hasattr(item, "path") else item.fspath)

            # Check if this test is under a superseded test path
            if not any(tp in item_path for tp in test_paths):
                continue

            # If no exchanges filter, skip everything under the path
            if not exchanges:
                item.add_marker(skip_marker)
                continue

            # Skip only if the test file matches a superseded exchange
            for exchange in exchanges:
                if f"{exchange}_" in item_path or f"/{exchange}/" in item_path:
                    item.add_marker(skip_marker)
                    break
