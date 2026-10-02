"""Overlay shim parity: hummingbot.core.api_throttler must resolve to web_assistant.throttler."""

import importlib

import pytest

MAPPED = {
    "hummingbot.core.api_throttler.async_request_context_base": "web_assistant.throttler.async_request_context_base",
    "hummingbot.core.api_throttler.async_throttler": "web_assistant.throttler.async_throttler",
    "hummingbot.core.api_throttler.async_throttler_base": "web_assistant.throttler.async_throttler_base",
    "hummingbot.core.api_throttler.data_types": "web_assistant.throttler.data_types",
    "hummingbot.core.web_assistant.auth": "web_assistant.auth",
    "hummingbot.core.web_assistant.connections": "web_assistant.connections",
    "hummingbot.core.web_assistant.connections.connections_factory": "web_assistant.connections.connections_factory",
    "hummingbot.core.web_assistant.connections.data_types": "web_assistant.connections.data_types",
    "hummingbot.core.web_assistant.connections.rest_connection": "web_assistant.connections.rest_connection",
    "hummingbot.core.web_assistant.connections.ws_connection": "web_assistant.connections.ws_connection",
    "hummingbot.core.web_assistant.connections.ws_data_types": "web_assistant.connections.ws_data_types",
    "hummingbot.core.web_assistant.rest_assistant": "web_assistant.rest_assistant",
    "hummingbot.core.web_assistant.rest_post_processors": "web_assistant.rest_post_processors",
    "hummingbot.core.web_assistant.rest_pre_processors": "web_assistant.rest_pre_processors",
    "hummingbot.core.web_assistant.web_assistants_factory": "web_assistant.web_assistants_factory",
    "hummingbot.core.web_assistant.ws_assistant": "web_assistant.ws_assistant",
    "hummingbot.core.web_assistant.ws_post_processors": "web_assistant.ws_post_processors",
    "hummingbot.core.web_assistant.ws_pre_processors": "web_assistant.ws_pre_processors",
}


@pytest.mark.parametrize(("legacy", "target"), sorted(MAPPED.items()))
def test_legacy_path_resolves_to_subpackage_module(legacy: str, target: str) -> None:
    mod = importlib.import_module(legacy)
    assert mod is importlib.import_module(target)
    assert mod.__name__ == target
