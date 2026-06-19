"""Conftest: remove compiled event_listener artifacts before test collection.

This runs at conftest import time (before any tests are collected or modules
are imported by test files), ensuring that when test_event_listener_py.py
imports EventListener it gets the .py version rather than the compiled .so.
"""

import pathlib
import sys

_BASE = pathlib.Path(__file__).parents[4]  # worktree root
_EVENT_DIR = _BASE / "hummingbot" / "core" / "event"

# All artifacts for the C1 conversion that must be removed.
_REMOVE = [
    *_EVENT_DIR.glob("event_listener.cpython-*.so"),
    _EVENT_DIR / "event_listener.pyx",
    _EVENT_DIR / "event_listener.pxd",
    _EVENT_DIR / "event_listener.cpp",
]

for _p in _REMOVE:
    if _p.exists():
        _p.unlink()

# Evict any already-loaded compiled module from sys.modules so pytest gets .py
for _key in list(sys.modules):
    if "event_listener" in _key:
        del sys.modules[_key]
