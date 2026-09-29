"""Overlay shim parity: hummingbot.core.api_throttler must resolve to web_assistant.throttler."""

import importlib

import pytest

MAPPED = {
    "hummingbot.core.api_throttler.async_request_context_base": "web_assistant.throttler.async_request_context_base",
    "hummingbot.core.api_throttler.async_throttler": "web_assistant.throttler.async_throttler",
    "hummingbot.core.api_throttler.async_throttler_base": "web_assistant.throttler.async_throttler_base",
    "hummingbot.core.api_throttler.data_types": "web_assistant.throttler.data_types",
}


@pytest.mark.parametrize(("legacy", "target"), sorted(MAPPED.items()))
def test_legacy_path_resolves_to_subpackage_module(legacy: str, target: str) -> None:
    mod = importlib.import_module(legacy)
    assert mod is importlib.import_module(target)
    assert mod.__name__ == target
