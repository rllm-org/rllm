def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


# Import environment classes
ENV_CLASSES = {
    "tool": safe_import("rllm.environments.tools.tool_env", "ToolEnvironment"),
}

# Import agent classes
AGENT_CLASSES = {
    "tool_agent": safe_import("rllm.agents.tool_agent", "ToolAgent"),
}

WORKFLOW_CLASSES = {
    "single_turn_workflow": safe_import("rllm.workflows.single_turn_workflow", "SingleTurnWorkflow"),
    "multi_turn_workflow": safe_import("rllm.workflows.multi_turn_workflow", "MultiTurnWorkflow"),
    "simple_workflow": safe_import("rllm.workflows.simple_workflow", "SimpleWorkflow"),
    "cumulative_workflow": safe_import("rllm.workflows.cumulative_workflow", "CumulativeWorkflow"),
}

# Filter out None values for unavailable imports
ENV_CLASS_MAPPING = {k: v for k, v in ENV_CLASSES.items() if v is not None}
AGENT_CLASS_MAPPING = {k: v for k, v in AGENT_CLASSES.items() if v is not None}
WORKFLOW_CLASS_MAPPING = {k: v for k, v in WORKFLOW_CLASSES.items() if v is not None}
