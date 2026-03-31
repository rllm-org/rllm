import os

import pytest

AGENT_ARN = os.environ.get("AGENTCORE_AGENT_ARN")
S3_BUCKET = os.environ.get("AGENTCORE_S3_BUCKET")
BASE_URL = os.environ.get("AGENTCORE_BASE_URL")
MODEL_ID = os.environ.get("AGENTCORE_MODEL_ID")

requires_agentcore = pytest.mark.skipif(
    not (AGENT_ARN and S3_BUCKET and BASE_URL and MODEL_ID),
    reason="AGENTCORE_AGENT_ARN, AGENTCORE_S3_BUCKET, AGENTCORE_BASE_URL, and AGENTCORE_MODEL_ID env vars required",
)

TINKER_API_KEY = os.environ.get("TINKER_API_KEY")
TINKER_MODEL_NAME = "Qwen/Qwen3-8B"

requires_tinker = pytest.mark.skipif(
    not TINKER_API_KEY,
    reason="TINKER_API_KEY env var required",
)