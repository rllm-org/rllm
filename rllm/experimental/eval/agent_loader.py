"""Agent loader: resolves agent by registry name, import path, or entry point."""

from __future__ import annotations

import importlib
import json
import os
from importlib.metadata import entry_points

from rllm.experimental.eval.types import AgentFlow

_RLLM_HOME = os.environ.get("RLLM_HOME", os.path.expanduser("~/.rllm"))
_USER_AGENTS_FILE = os.path.join(_RLLM_HOME, "agents.json")


def _load_user_agents() -> dict[str, dict]:
    """Load the user-registered agents from ~/.rllm/agents.json."""
    if not os.path.exists(_USER_AGENTS_FILE):
        return {}
    try:
        with open(_USER_AGENTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_user_agents(registry: dict[str, dict]) -> None:
    """Persist the user-registered agents to ~/.rllm/agents.json."""
    os.makedirs(os.path.dirname(_USER_AGENTS_FILE), exist_ok=True)
    with open(_USER_AGENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def register_agent(name: str, agent_or_path: AgentFlow | type | str) -> None:
    """Persist an agent registration so it's discoverable by name across processes.

    The registration is saved to ``~/.rllm/agents.json``.  After calling this,
    ``load_agent(name)`` and ``rllm eval --agent <name>`` will resolve the agent
    from a new Python process or CLI session.

    Args:
        name: Short name for the agent (e.g., ``"concierge"``).
        agent_or_path: One of:
            - An import-path string (e.g., ``"my_agent.agent:ConciergeAgent"``).
            - A class — the import path is derived automatically.
            - An instance — the import path is derived from its class.

    Raises:
        TypeError: If an instance/class is provided but doesn't have ``.run()``.
    """
    if isinstance(agent_or_path, str):
        import_path = agent_or_path
    elif isinstance(agent_or_path, type):
        _validate_agent_class(agent_or_path, name)
        import_path = f"{agent_or_path.__module__}:{agent_or_path.__qualname__}"
    else:
        _validate_agent(agent_or_path, name)
        cls = type(agent_or_path)
        import_path = f"{cls.__module__}:{cls.__qualname__}"

    registry = _load_user_agents()
    registry[name] = {"import_path": import_path}
    _save_user_agents(registry)


def unregister_agent(name: str) -> bool:
    """Remove a user-registered agent.

    Returns:
        True if the agent was found and removed, False otherwise.
    """
    registry = _load_user_agents()
    if name not in registry:
        return False
    del registry[name]
    _save_user_agents(registry)
    return True


def _load_agent_catalog() -> dict:
    """Load the agents.json catalog from the registry directory."""
    catalog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "registry", "agents.json")
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def _import_from_path(import_path: str) -> object:
    """Import an object from a 'module:attribute' path."""
    module_path, attr_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _validate_agent(obj: object, name: str) -> None:
    """Validate that an object conforms to the AgentFlow protocol."""
    if not hasattr(obj, "run") or not callable(obj.run):
        raise TypeError(f"Agent '{name}' must be an AgentFlow with a .run() method, got {type(obj).__name__}")


def _validate_agent_class(cls: type, name: str) -> None:
    """Validate that a class has a .run method."""
    run_attr = getattr(cls, "run", None)
    if run_attr is None or not callable(run_attr):
        raise TypeError(f"Agent '{name}' must be an AgentFlow class with a .run() method, got {cls.__name__}")


def _load_and_instantiate(import_path: str, name: str) -> AgentFlow:
    """Import an agent from a path, auto-instantiate classes, and validate."""
    obj = _import_from_path(import_path)
    if isinstance(obj, type):
        obj = obj()
    _validate_agent(obj, name)
    return obj


def load_agent(name_or_path: str) -> AgentFlow:
    """Load an agent by registry name, import path, or entry point.

    Lookup order: user registry (``~/.rllm/agents.json``) → colon import path
    → built-in catalog (``registry/agents.json``) → entry points
    (``rllm.agents`` group).

    Args:
        name_or_path: A registry name (e.g., ``"math"``), a colon-separated
            import path (e.g., ``"my_module:my_agent"``), or a plugin name
            registered via the ``rllm.agents`` entry-point group.

    Returns:
        An AgentFlow instance with a ``.run()`` method.
    """
    # 1. User-registered agents (persistent, from register_agent())
    user_agents = _load_user_agents()
    if name_or_path in user_agents:
        return _load_and_instantiate(user_agents[name_or_path]["import_path"], name_or_path)

    # 2. Explicit import path: "my_module:my_agent"
    if ":" in name_or_path:
        return _load_and_instantiate(name_or_path, name_or_path)

    # 3. Built-in catalog
    catalog = _load_agent_catalog()
    agents = catalog.get("agents", {})
    if name_or_path in agents:
        entry = agents[name_or_path]
        module = importlib.import_module(entry["module"])
        obj = getattr(module, entry["function"])
        _validate_agent(obj, name_or_path)
        return obj

    # 4. Plugin discovery via entry points
    eps = entry_points(group="rllm.agents")
    for ep in eps:
        if ep.name == name_or_path:
            obj = ep.load()
            if isinstance(obj, type):
                obj = obj()
            _validate_agent(obj, name_or_path)
            return obj

    available = ", ".join(sorted(agents.keys()))
    raise KeyError(f"Agent '{name_or_path}' not found in registry or agenthub. Available built-in: {available}")


def list_agents() -> list[dict]:
    """List all available agents (user-registered + built-in + agenthub).

    Returns:
        A list of dicts with keys: name, source, description, module.
    """
    results: list[dict] = []

    # User-registered agents
    for name, info in sorted(_load_user_agents().items()):
        results.append(
            {
                "name": name,
                "source": "registered",
                "description": "",
                "module": info["import_path"],
            }
        )

    # Built-in agents from catalog
    seen_names = {r["name"] for r in results}
    try:
        catalog = _load_agent_catalog()
    except (FileNotFoundError, json.JSONDecodeError):
        catalog = {}

    for name, info in sorted(catalog.get("agents", {}).items()):
        if name not in seen_names:
            results.append(
                {
                    "name": name,
                    "source": "built-in",
                    "description": info.get("description", ""),
                    "module": f"{info.get('module', '')}.{info.get('function', '')}",
                }
            )

    # Plugin agents from entry points
    seen_names = {r["name"] for r in results}
    eps = entry_points(group="rllm.agents")
    for ep in eps:
        if ep.name not in seen_names:
            pkg = ep.dist.name if ep.dist else "unknown"
            results.append(
                {
                    "name": ep.name,
                    "source": f"plugin ({pkg})",
                    "description": "",
                    "module": str(ep.value),
                }
            )

    return results
