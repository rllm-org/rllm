import os

import pytest

TINKER_API_KEY = os.environ.get("TINKER_API_KEY")
TINKER_MODEL_NAME = "Qwen/Qwen3-8B"

requires_tinker = pytest.mark.skipif(
    not TINKER_API_KEY,
    reason="TINKER_API_KEY env var required",
)
