[project]
name = "{{project_name}}"
version = "0.1.0"
description = "{{description}}"
requires-python = ">=3.10"
dependencies = [
    "rllm",
    "openai",
{{extra_deps}}]

# Register your agent/evaluator as plugins so `rllm agent list` discovers them.
# After `pip install -e .`, rllm will find these automatically — no manual
# registration needed.
[project.entry-points."rllm.agents"]
{{agent_name}} = "{{module_name}}.agent:{{agent_instance}}"

# Uncomment to register a custom evaluator as a plugin:
# [project.entry-points."rllm.evaluators"]
# {{agent_name}} = "{{module_name}}.evaluator:{{evaluator_class}}"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"
