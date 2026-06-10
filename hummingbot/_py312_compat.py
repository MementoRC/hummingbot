"""Python 3.12 stdlib-removal compatibility shims for the bleeding-edge fork.

Owned by `_for_bleed/py312-runtime` (infrastructure tier). Provides runtime
fallbacks for stdlib modules removed in Python 3.12 / 3.13 so that imports of
those modules (rewritten by `RemovedStdlibImportRule` in the py312_rules
package) raise a clear, actionable error at use time rather than producing a
cryptic AttributeError on first attribute access.

The transform rule rewrites:

    import asynchat
    asynchat.async_chat()

to:

    asynchat = _Py312RemovedModule("asynchat")
    asynchat.async_chat()    # raises Py312RemovedModuleError on attr access

Files using such a shim must import this helper. The transform rule is
expected to inject the import at the top of every modified file in a future
revision; until then, manual import of `_Py312RemovedModule` is required for
files that the rule rewrites.
"""

from __future__ import annotations

from typing import NoReturn

REMOVED_IN_312 = {
    "asynchat",
    "asyncore",
    "smtpd",
    "sndhdr",
    "telnetlib",
    "imghdr",
    "mailcap",
    "nntplib",
    "ossaudiodev",
    "spwd",
    "xdrlib",
}


class Py312RemovedModuleError(ImportError):
    """Raised when code attempts to use a stdlib module removed in Python 3.12+."""


class _Py312RemovedModule:
    """Lazy raise-on-access proxy for a removed stdlib module.

    Constructed once at import-rewrite time. Any attribute access raises
    `Py312RemovedModuleError` with a message naming the missing module — making
    the migration owner aware of which call site needs to be replaced with the
    appropriate successor (e.g. `asynchat` → `aiohttp`/`asyncio`-based
    handlers; `smtpd` → `aiosmtpd`).
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> NoReturn:
        raise Py312RemovedModuleError(
            f"stdlib module '{self._name}' was removed in Python 3.12 — "
            f"attempted to access '{self._name}.{attr}'. "
            f"Replace this call site with the module's documented successor."
        )

    def __repr__(self) -> str:
        return f"<_Py312RemovedModule {self._name!r} (use raises Py312RemovedModuleError)>"

    def __bool__(self) -> bool:
        return False


def _module_was_removed_in_312(name: str) -> bool:
    """Helper for static analysis tools to query the removed-set."""
    return name in REMOVED_IN_312
