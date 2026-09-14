"""GAIA search agent for `rllm eval gaia`.

The search + browse ReAct agent now ships as the built-in ``search`` harness
(``rllm.harnesses.search:SearchHarness``) so ``rllm eval gaia`` works without
an explicit ``--agent``. This module re-exports it for backward compatibility
with ``--agent cookbooks.gaia.agent:agent``.

Run:
    export TAVILY_API_KEY=...      # web search + page extract
    export HF_TOKEN=...            # GAIA dataset is gated
    rllm model setup               # configure your model provider
    rllm eval gaia --max-examples 5
"""

from rllm.harnesses.search import (
    AGENT_SYSTEM_PROMPT as AGENT_SYSTEM_PROMPT,
)
from rllm.harnesses.search import (
    MAX_TURNS as MAX_TURNS,
)
from rllm.harnesses.search import (
    SearchHarness as SearchHarness,
)
from rllm.harnesses.search import (
    run_tool_loop as run_tool_loop,
)

# Backward-compat alias for `--agent cookbooks.gaia.agent:agent`
GaiaAgent = SearchHarness

agent = GaiaAgent()
