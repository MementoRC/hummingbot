"""Core contract tests: concrete implementations vs core-defined ports."""

from __future__ import annotations

import inspect
from typing import Any


def _protocol_members(protocol: type) -> dict[str, Any]:
    """Return the members a Protocol declares (excluding dunder/protocol plumbing)."""
    members: dict[str, Any] = {}
    for klass in reversed(protocol.__mro__):
        if klass is object or klass.__name__ in ("Protocol", "Generic"):
            continue
        for name, value in vars(klass).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            if name.startswith("_abc_") or name in ("_is_protocol", "_is_runtime_protocol"):
                continue
            members[name] = value
        for name in getattr(klass, "__annotations__", {}):
            members.setdefault(name, None)
    return members


def assert_conforms(obj: Any, protocol: type) -> None:
    """Assert that ``obj`` structurally conforms to ``protocol``.

    Checks that every protocol member exists on ``obj``, that callable members are
    callable on ``obj``, and that sync/async-ness matches (``inspect.iscoroutinefunction``).
    Raises ``AssertionError`` listing every violation.
    """
    problems: list[str] = []
    for name, expected in _protocol_members(protocol).items():
        if not hasattr(obj, name):
            problems.append(f"missing member '{name}'")
            continue
        if not callable(expected):
            continue
        actual = getattr(obj, name)
        if not callable(actual):
            problems.append(f"member '{name}' is not callable")
        elif inspect.iscoroutinefunction(expected) != inspect.iscoroutinefunction(actual):
            want = "async" if inspect.iscoroutinefunction(expected) else "sync"
            problems.append(f"member '{name}' must be {want}")
    if problems:
        raise AssertionError(f"{type(obj).__name__} does not conform to {protocol.__name__}: " + "; ".join(problems))
