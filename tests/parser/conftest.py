"""Parser tests require real packages (transformers, pydantic, torch, etc.).

The root conftest.py stubs out heavy optional dependencies for lightweight unit
tests. This conftest removes the specific stubs so parser integration tests can
use real packages.
"""

import sys
import types

# These are the exact modules stubbed by root conftest.py _STUB_MODULES list,
# plus the additional stubs it creates for sub-modules and fake classes.
_ROOT_STUB_MODULES = [
    "numpy",
    "httpx",
    "transformers",
    "datasets",
    "ray",
    "pandas",
    "polars",
    "sympy",
    "pylatexenc",
    "antlr4",
    "antlr4_python3_runtime",
    "mcp",
    "eval_protocol",
    "hydra",
    "fastapi",
    "uvicorn",
    "tqdm",
    "yaml",
    "pydantic",
    "wrapt",
    "asgiref",
    "wandb",
    "codetiming",
    "click",
    # Also stubbed explicitly by root conftest
    "torch",
    "PIL",
    "openai",
]

# Remove stub modules and any sub-modules created by root conftest
_to_remove = []
for name in list(sys.modules):
    base = name.split(".")[0]
    if base in _ROOT_STUB_MODULES:
        mod = sys.modules[name]
        if isinstance(mod, types.ModuleType) and not hasattr(mod, "__file__"):
            _to_remove.append(name)

for name in _to_remove:
    del sys.modules[name]
