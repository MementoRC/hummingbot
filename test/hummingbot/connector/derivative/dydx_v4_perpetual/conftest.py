import importlib.util

import pytest

if importlib.util.find_spec("v4_proto") is None:
    pytest.skip("v4_proto not installed", allow_module_level=True)
