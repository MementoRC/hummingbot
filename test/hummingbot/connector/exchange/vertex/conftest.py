import importlib.util

import pytest

if importlib.util.find_spec("eip712_structs") is None:
    pytest.skip("eip712_structs not installed", allow_module_level=True)
