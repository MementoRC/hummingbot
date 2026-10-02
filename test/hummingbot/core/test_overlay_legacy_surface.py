"""Overlay shim surface parity: names a legacy module defined must exist on the module it now resolves to."""

import importlib
import json
from pathlib import Path
import unittest

from test.hummingbot.core.api_throttler.test_overlay_shim import MAPPED

SNAPSHOT_PATH = Path(__file__).parent / "overlay_legacy_surface.json"


def _load_snapshot() -> dict[str, list[str]]:
    with SNAPSHOT_PATH.open() as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


class OverlayLegacySurfaceTests(unittest.TestCase):
    def test_snapshot_keys_are_mapped(self):
        unmapped = sorted(set(_load_snapshot()) - set(MAPPED))
        self.assertEqual([], unmapped, f"Snapshot modules missing from MAPPED: {unmapped}")

    def test_legacy_public_names_present_on_target(self):
        for legacy, names in sorted(_load_snapshot().items()):
            with self.subTest(module=legacy):
                mod = importlib.import_module(legacy)
                target_public = {n for n in dir(mod) if not n.startswith("_")}
                missing = sorted(set(names) - target_public)
                self.assertEqual([], missing, f"{legacy} -> {mod.__name__} is missing legacy names: {missing}")
