"""Root conftest: Phase-C artifact cleanup + sub-package supersede skip.

PHASE-C CLEANUP  (runs at conftest import time, before any test collection)
---------------
EventListener (C1) was converted to pure Python.  Any compiled .so built
against the old Cython cdef-class EventListener (48-byte struct) fails with
"size changed, may indicate binary incompatibility" at import time once
event_listener.so is removed.  The modules below already have pure-Python
.py replacements written in earlier Phase-C steps.  Removing the stale .so
files here lets Python fall through to the .py versions for the test session.

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
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# ---------------------------------------------------------------------------
# Phase-C: remove stale compiled artifacts that encode the old Cython
# EventListener struct size (48 bytes).  Pure-Python .py replacements exist
# for every module listed here.
# ---------------------------------------------------------------------------
_HBOT = Path(__file__).parents[1] / "hummingbot"
_PHASE_C_ORPHANED_SO = [
    # C1 — EventListener base class (this conversion)
    _HBOT / "core/event/event_listener.cpython-312-x86_64-linux-gnu.so",
    # C2 — EventLogger
    _HBOT / "core/event/event_logger.cpython-312-x86_64-linux-gnu.so",
    # C3 — EventReporter
    _HBOT / "core/event/event_reporter.cpython-312-x86_64-linux-gnu.so",
    # C4 — NetworkIterator
    _HBOT / "core/network_iterator.cpython-312-x86_64-linux-gnu.so",
    # C5 — TimeIterator
    _HBOT / "core/time_iterator.cpython-312-x86_64-linux-gnu.so",
    # C6 — OrderBook (pure-Python .py already written)
    _HBOT / "core/data_type/order_book.cpython-312-x86_64-linux-gnu.so",
    # C7/C8 — ConnectorBase (pure-Python .py already written)
    _HBOT / "connector/connector_base.cpython-312-x86_64-linux-gnu.so",
    # CompositeOrderBook (pure-Python .py now written as part of C1 cascade fix)
    _HBOT / "core/data_type/composite_order_book.cpython-312-x86_64-linux-gnu.so",
]
for _so in _PHASE_C_ORPHANED_SO:
    if _so.exists():
        _so.unlink()

# Evict any module that may have been cached before this conftest ran.
_EVICT = {
    "event_listener",
    "event_logger",
    "event_reporter",
    "network_iterator",
    "time_iterator",
    "order_book",
    "composite_order_book",
    "connector_base",
}
for _key in list(sys.modules):
    if any(k in _key for k in _EVICT):
        del sys.modules[_key]

# ---------------------------------------------------------------------------
# Phase-C: rebuild pubsub.so against the new Python EventListener.
#
# pubsub.pyx no longer `cimport`s EventListener (the cimport was removed as
# part of C1).  The old pubsub.so still embeds sizeof(EventListener)==48 from
# compile time.  We must recompile before any test imports PubSub.
# ---------------------------------------------------------------------------
# Cython modules that encode the old EventListener struct size and must be
# rebuilt.  None of these have pure-Python replacements yet.
_CYTHON_REBUILD = [
    # (module_name, pyx_path, so_path, cpp_path)
    (
        "hummingbot.core.pubsub",
        _HBOT / "core/pubsub.pyx",
        _HBOT / "core/pubsub.cpython-312-x86_64-linux-gnu.so",
        _HBOT / "core/pubsub.cpp",
    ),
    (
        "hummingbot.core.clock",
        _HBOT / "core/clock.pyx",
        _HBOT / "core/clock.cpython-312-x86_64-linux-gnu.so",
        _HBOT / "core/clock.cpp",
    ),
    (
        "hummingbot.core.py_time_iterator",
        _HBOT / "core/py_time_iterator.pyx",
        _HBOT / "core/py_time_iterator.cpython-312-x86_64-linux-gnu.so",
        _HBOT / "core/py_time_iterator.cpp",
    ),
]


def _rebuild_cython_modules() -> None:
    """Recompile Cython modules that were built against the old EventListener."""
    import os
    import subprocess
    import tempfile

    repo_root = Path(__file__).parents[1]

    modules_to_build = []
    for mod_name, pyx, so, cpp in _CYTHON_REBUILD:
        if not pyx.exists():
            continue
        # Remove stale artifacts.
        for stale in (so, cpp):
            if stale.exists():
                stale.unlink()
        # Evict from sys.modules.
        for key in list(sys.modules):
            if mod_name.split(".")[-1] in key:
                del sys.modules[key]
        modules_to_build.append((mod_name, str(pyx)))

    if not modules_to_build:
        return

    # Build all stale modules in one compiler invocation.
    ext_defs = "\n".join(
        f'    Extension("{name}", sources=["{pyx}"], language="c++",'
        f' extra_compile_args=["-std=c++11"], extra_link_args=["-std=c++11"],'
        f" include_dirs=[np.get_include()]),"
        for name, pyx in modules_to_build
    )
    script_src = f"""\
import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension

exts = cythonize(
    [
{ext_defs}
    ],
    language_level=3,
    language="c++",
    compiler_directives={{"annotation_typing": False}},
)
setup(name="hb_phase_c_rebuild", ext_modules=exts)
"""
    build_script = str(repo_root / "_phase_c_rebuild.py")
    with open(build_script, "w") as f:
        f.write(script_src)

    try:
        with tempfile.TemporaryDirectory() as build_tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    build_script,
                    "build_ext",
                    "--inplace",
                    "--build-lib",
                    str(repo_root / "build"),
                    "--build-temp",
                    build_tmp,
                ],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
            )
        if result.returncode != 0:
            raise RuntimeError(f"Phase-C Cython rebuild failed (exit {result.returncode}):\n{result.stderr[-2000:]}")
    finally:
        if os.path.exists(build_script):
            os.remove(build_script)


_rebuild_cython_modules()


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
