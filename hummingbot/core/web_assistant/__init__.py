"""Compat shim: hummingbot.core.web_assistant -> web_assistant sub-package.

The web_assistant sub-package was extracted from hummingbot.core.web_assistant.
This shim registers sys.modules aliases so that legacy code and tests that
patch 'hummingbot.core.web_assistant.*' paths target the same module objects
as the actual web_assistant package.
"""

import importlib
import sys

# Eagerly import all web_assistant submodules that legacy tests reference.
# This ensures sys.modules is populated before we alias.
_SUBMODULES = [
    "web_assistant.auth",
    "web_assistant.connections",
    "web_assistant.connections.connections_factory",
    "web_assistant.connections.data_types",
    "web_assistant.connections.rest_connection",
    "web_assistant.connections.ws_connection",
    "web_assistant.connections.ws_data_types",
    "web_assistant.rest_assistant",
    "web_assistant.rest_post_processors",
    "web_assistant.rest_pre_processors",
    "web_assistant.throttler",
    "web_assistant.throttler.async_throttler",
    "web_assistant.web_assistants_factory",
    "web_assistant.ws_assistant",
    "web_assistant.ws_post_processors",
    "web_assistant.ws_pre_processors",
]

for _mod in _SUBMODULES:
    try:
        importlib.import_module(_mod)
    except ImportError:
        pass  # sub-module may not exist in all versions; skip gracefully

# Register hummingbot.core.web_assistant.* aliases for every web_assistant.*
# module currently in sys.modules.  Must happen after the eager imports above.
_PREFIX = "web_assistant"
_ALIAS_PREFIX = "hummingbot.core.web_assistant"

for _key, _mod_obj in list(sys.modules.items()):
    if _key == _PREFIX or _key.startswith(_PREFIX + "."):
        _alias = _ALIAS_PREFIX + _key[len(_PREFIX) :]
        sys.modules.setdefault(_alias, _mod_obj)

# Make this package itself an alias for web_assistant.
import web_assistant as _wa  # noqa: E402

sys.modules[_ALIAS_PREFIX] = _wa
