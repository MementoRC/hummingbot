"""Compat shim: hummingbot.core.web_assistant -> web_assistant (hb-web-assistant).

Every legacy module is registered in sys.modules under its legacy dotted name,
packages and leaves alike. Importing any ``hummingbot.core.web_assistant.X.Y``
first executes this ``__init__`` (parent packages always import before
children), so the registrations below are in place before the import system
looks up the child name and finds the target module object already there.
"""

import sys

import web_assistant.auth as _auth
import web_assistant.connections as _connections
import web_assistant.connections.connections_factory as _connections_factory
import web_assistant.connections.data_types as _data_types
import web_assistant.connections.rest_connection as _rest_connection
import web_assistant.connections.ws_connection as _ws_connection
import web_assistant.connections.ws_data_types as _ws_data_types
import web_assistant.rest_assistant as _rest_assistant
import web_assistant.rest_post_processors as _rest_post_processors
import web_assistant.rest_pre_processors as _rest_pre_processors
import web_assistant.web_assistants_factory as _web_assistants_factory
import web_assistant.ws_assistant as _ws_assistant
import web_assistant.ws_post_processors as _ws_post_processors
import web_assistant.ws_pre_processors as _ws_pre_processors

_PREFIX = "hummingbot.core.web_assistant"

sys.modules[f"{_PREFIX}.auth"] = _auth
sys.modules[f"{_PREFIX}.connections"] = _connections
sys.modules[f"{_PREFIX}.connections.connections_factory"] = _connections_factory
sys.modules[f"{_PREFIX}.connections.data_types"] = _data_types
sys.modules[f"{_PREFIX}.connections.rest_connection"] = _rest_connection
sys.modules[f"{_PREFIX}.connections.ws_connection"] = _ws_connection
sys.modules[f"{_PREFIX}.connections.ws_data_types"] = _ws_data_types
sys.modules[f"{_PREFIX}.rest_assistant"] = _rest_assistant
sys.modules[f"{_PREFIX}.rest_post_processors"] = _rest_post_processors
sys.modules[f"{_PREFIX}.rest_pre_processors"] = _rest_pre_processors
sys.modules[f"{_PREFIX}.web_assistants_factory"] = _web_assistants_factory
sys.modules[f"{_PREFIX}.ws_assistant"] = _ws_assistant
sys.modules[f"{_PREFIX}.ws_post_processors"] = _ws_post_processors
sys.modules[f"{_PREFIX}.ws_pre_processors"] = _ws_pre_processors
