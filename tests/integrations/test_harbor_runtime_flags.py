"""HarborRuntime must ask the engine for the publicly-reachable gateway URL.

Harbor installed agents run their LLM client inside the sandbox. The engine
keys this off ``llm_inside_env`` (agentflow_engine.py); without it the agent
env gets the host-local URL with the docker-only ``host.docker.internal``
rewrite, which does not resolve on modal/daytona/e2b — every task fails its
first LLM call with "OpenAIException - Connection error".
"""

from rllm.integrations.harbor.runtime import HarborRuntime


def test_harbor_agents_declare_llm_inside_env():
    assert HarborRuntime.llm_inside_env is True
    assert HarborRuntime(agent_name="openhands-sdk").llm_inside_env is True
