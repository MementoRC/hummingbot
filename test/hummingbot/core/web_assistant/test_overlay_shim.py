"""Overlay shim parity: hummingbot.core.web_assistant must resolve to the web_assistant sub-package."""

import importlib

import pytest

_LEGACY = "hummingbot.core.web_assistant."
_TARGET = "web_assistant."

_NAMES = [
    "auth",
    "rest_assistant",
    "rest_post_processors",
    "rest_pre_processors",
    "web_assistants_factory",
    "ws_assistant",
    "ws_post_processors",
    "ws_pre_processors",
    "connections.connections_factory",
    "connections.data_types",
    "connections.rest_connection",
    "connections.ws_connection",
]

MAPPED = {_LEGACY + name: _TARGET + name for name in _NAMES}

WS_CLASSES = ["WSRequest", "WSJSONRequest", "WSPlainTextRequest", "WSBinaryRequest", "WSResponse"]


@pytest.mark.parametrize(("legacy", "target"), sorted(MAPPED.items()))
def test_legacy_path_resolves_to_subpackage_module(legacy: str, target: str) -> None:
    mod = importlib.import_module(legacy)
    assert mod is importlib.import_module(target)
    assert mod.__name__ == target


@pytest.mark.parametrize(("legacy", "target"), sorted(MAPPED.items()))
def test_public_names_preserved(legacy: str, target: str) -> None:
    # The in-tree file (still present before the shim lands) defines the contract.
    legacy_mod = importlib.import_module(legacy)
    target_mod = importlib.import_module(target)
    legacy_public = {n for n in dir(legacy_mod) if not n.startswith("_")}
    target_public = {n for n in dir(target_mod) if not n.startswith("_")}
    assert legacy_public <= target_public, sorted(legacy_public - target_public)


@pytest.mark.parametrize("name", WS_CLASSES)
def test_ws_class_identity_across_data_types_modules(name: str) -> None:
    data_types = importlib.import_module("web_assistant.connections.data_types")
    ws_data_types = importlib.import_module("web_assistant.connections.ws_data_types")
    assert getattr(data_types, name) is getattr(ws_data_types, name)
