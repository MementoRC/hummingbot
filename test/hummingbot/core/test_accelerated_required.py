"""Guard that the Rust extension is really in use when HB_REQUIRE_ACCELERATED=1.

Without the env var the tests are skipped, so Python-fallback environments stay green.
"""

import importlib
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("HB_REQUIRE_ACCELERATED") != "1",
    reason="HB_REQUIRE_ACCELERATED=1 not set",
)


@pytest.mark.parametrize(
    "module_path",
    [
        "hummingbot.core.rust_metrics",
        "hummingbot.core.data_type.order_expiration_entry_rust",
    ],
)
def test_module_is_accelerated(module_path: str) -> None:
    mod = importlib.import_module(module_path)
    accelerated = getattr(mod, "_ACCELERATED")  # noqa: B009
    assert accelerated is True, f"{module_path}: _ACCELERATED is False (pure-Python fallback)"
