from __future__ import annotations

import os

import pytest
import requests

from rllm.integrations.mcp_atlas.constants import IMAGE
from rllm.integrations.mcp_atlas.service import MCPAtlasServiceManager

pytestmark = pytest.mark.skipif(
    os.environ.get("MCP_ATLAS_LIVE") != "1",
    reason="set MCP_ATLAS_LIVE=1 and the MCP_ATLAS_LIVE_LLM_* variables to run the Docker contract probe",
)


def test_official_customer_probe_against_pinned_image(tmp_path):
    base_url = os.environ["MCP_ATLAS_LIVE_LLM_BASE_URL"]
    api_key = os.environ["MCP_ATLAS_LIVE_LLM_API_KEY"]
    model = os.environ.get("MCP_ATLAS_LIVE_MODEL", "openai/gpt-4o")
    manager = MCPAtlasServiceManager(
        image=IMAGE,
        preflight="smoke",
        required_servers={"filesystem"},
        run_dir=tmp_path,
    )
    try:
        manager.start()
        response = requests.post(
            f"{manager.harness_url}/v2/mcp_eval/run_agent",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "What is the first word of the file at /data/Barber Shop.csv?"}],
                "enabledTools": ["filesystem_read_text_file"],
                "image": IMAGE,
                "llm_base_url": base_url,
                "extra_llm_params": {"api_key": api_key},
                "max_turns": 256,
                "max_tool_calls": 100,
            },
            timeout=300,
        )
        response.raise_for_status()
        messages = [event["data"] for event in response.json() if event.get("type") == "message"]
        final = next(message["content"] for message in reversed(messages) if message.get("role") == "assistant" and message.get("content"))
        assert "Customer" in final
    finally:
        manager.stop()
