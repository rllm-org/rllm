import importlib
import sys
import types


class _FakeTensor(list):
    """A minimal stand-in for torch.Tensor for unit tests."""


class _FakeTorch(types.ModuleType):
    long = "long"

    def tensor(self, data, dtype=None):  # noqa: ARG002
        return _FakeTensor(data)


# Provide a tiny torch stub so importing rllm modules doesn't require real torch.
# Only install the stub if the real torch package is NOT available.
if "torch" not in sys.modules:
    try:
        importlib.import_module("torch")
    except Exception:
        sys.modules["torch"] = _FakeTorch("torch")


# Provide a tiny PIL stub (OpenAIEngine imports PIL.Image)
# Only install the stub if the real PIL package is NOT available.
if "PIL" not in sys.modules:
    try:
        importlib.import_module("PIL")
    except Exception:
        pil = types.ModuleType("PIL")
        image_mod = types.ModuleType("PIL.Image")

        class Image:  # noqa: D401
            """Stub PIL.Image.Image."""

            pass

        image_mod.Image = Image
        pil.Image = image_mod
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = image_mod


# Provide a tiny openai stub (OpenAIEngine imports openai.AsyncOpenAI)
# Only install the stub if the real openai package is NOT available.
if "openai" not in sys.modules:
    try:
        importlib.import_module("openai")
    except Exception:
        openai = types.ModuleType("openai")

        class RateLimitError(Exception):
            pass

        class AsyncOpenAI:  # noqa: D401
            """Stub openai.AsyncOpenAI."""

            def __init__(self, *args, **kwargs):  # noqa: ANN001, ANN002
                self.args = args
                self.kwargs = kwargs

            class chat:  # noqa: D401
                class completions:  # noqa: D401
                    @staticmethod
                    async def create(*args, **kwargs):  # noqa: ANN001, ANN002
                        raise NotImplementedError

            class completions:  # noqa: D401
                @staticmethod
                async def create(*args, **kwargs):  # noqa: ANN001, ANN002
                    raise NotImplementedError

        class OpenAI:  # noqa: D401
            """Stub openai.OpenAI."""

            def __init__(self, *args, **kwargs):  # noqa: ANN001, ANN002
                self.args = args
                self.kwargs = kwargs

        openai.RateLimitError = RateLimitError
        openai.AsyncOpenAI = AsyncOpenAI
        openai.OpenAI = OpenAI
        sys.modules["openai"] = openai

        # Stub openai._models for rllm.sdk.chat.openai imports
        openai_models = types.ModuleType("openai._models")

        class _FinalRequestOptions:  # noqa: D401
            pass

        openai_models.FinalRequestOptions = _FinalRequestOptions
        sys.modules["openai._models"] = openai_models


# Many modules are optional / heavy (torch, transformers, ray, etc.).
# For lightweight unit tests here, stub them out so imports succeed.
_STUB_MODULES = [
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
]

for _name in _STUB_MODULES:
    if _name in sys.modules:
        continue
    try:
        importlib.import_module(_name)
    except Exception:
        sys.modules[_name] = types.ModuleType(_name)

# Provide minimal attributes expected by some modules during import
if hasattr(sys.modules.get("transformers"), "__dict__"):

    class PreTrainedTokenizerBase:  # noqa: D401
        pass

    sys.modules["transformers"].PreTrainedTokenizerBase = PreTrainedTokenizerBase

# pylatexenc is imported as `from pylatexenc import latex2text`
if "pylatexenc" in sys.modules and not hasattr(sys.modules["pylatexenc"], "latex2text"):
    latex2text = types.SimpleNamespace(LatexNodes2Text=lambda *a, **k: None)
    sys.modules["pylatexenc"].latex2text = latex2text

# sympy is imported with submodules (e.g. sympy.parsing.sympy_parser)
if "sympy" in sys.modules:
    sympy_mod = sys.modules["sympy"]
    parsing_mod = types.ModuleType("sympy.parsing")
    sympy_parser_mod = types.ModuleType("sympy.parsing.sympy_parser")
    parsing_mod.sympy_parser = sympy_parser_mod
    # expose modules
    sys.modules.setdefault("sympy.parsing", parsing_mod)
    sys.modules.setdefault("sympy.parsing.sympy_parser", sympy_parser_mod)
    # attach for attribute access
    sympy_mod.parsing = parsing_mod
