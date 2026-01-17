"""Utilities for extracting source code metadata from Python objects."""

import inspect
from typing import Any, Dict, Optional


def extract_source_metadata(
    workflow_class: Optional[Any] = None,
    agent_class: Optional[Any] = None,
    workflow_args: Optional[Dict] = None,
    env_args: Optional[Dict] = None,
) -> Dict[str, str]:
    """Extract source code metadata for UI logging.

    Extracts source code from workflow classes, agent classes, and reward functions
    for display in the training UI. Handles edge cases like lambda functions and
    objects where source code is unavailable.

    Args:
        workflow_class: Workflow class to extract source from.
        agent_class: Agent class to extract source from.
        workflow_args: Dict that may contain 'reward_function' key.
        env_args: Dict that may contain 'reward_fn' key.

    Returns:
        Dictionary containing source code and names for available components.
        Possible keys: workflow_class, workflow_source, agent_class, agent_source,
        reward_fn_name, reward_fn_source.
    """
    source_metadata: Dict[str, str] = {}

    def _extract_source(obj: Any, name_key: str, source_key: str, default_name: str) -> None:
        """Extract source code from a class or function."""
        if obj is None:
            return
        try:
            source_metadata[source_key] = inspect.getsource(obj)
            source_metadata[name_key] = obj.__name__
        except Exception:
            source_metadata[name_key] = getattr(obj, '__name__', default_name)
            if source_metadata[name_key] == '<lambda>':
                source_metadata[source_key] = "# Lambda function - source not available"

    # Extract workflow class source
    if workflow_class:
        _extract_source(workflow_class, 'workflow_class', 'workflow_source', 'Workflow')

    # Extract agent class source
    if agent_class:
        _extract_source(agent_class, 'agent_class', 'agent_source', 'Agent')

    # Extract reward function (from workflow_args or env_args)
    reward_fn = None
    if workflow_args:
        reward_fn = workflow_args.get('reward_function')
    if reward_fn is None and env_args:
        reward_fn = env_args.get('reward_fn')
    if reward_fn:
        _extract_source(reward_fn, 'reward_fn_name', 'reward_fn_source', 'reward_fn')

    return source_metadata
