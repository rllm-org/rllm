"""Top-level harnesses for rLLM.

Built-in agent flows live here. The CLI ``--agent`` flag picks one
by registered name; entries are listed in ``rllm/registry/agents.json``
and resolved through :func:`rllm.eval.agent_loader.load_agent`.
"""
