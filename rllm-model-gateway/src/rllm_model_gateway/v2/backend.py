from collections.abc import Callable
from typing import Any, Protocol

from rllm_model_gateway.v2.config import BackendConfig
from rllm_model_gateway.v2.contracts import TokenInput, TokenOutput


class GenerationBackend(Protocol):
    async def generate(self, request: TokenInput) -> TokenOutput: ...

    async def close(self) -> None: ...


BackendFactory = Callable[[dict[str, Any]], GenerationBackend]
BACKEND_FACTORIES: dict[str, BackendFactory] = {}


def build_backend(config: BackendConfig) -> GenerationBackend:
    factory = BACKEND_FACTORIES.get(config.name)
    if factory is None:
        raise ValueError(f"unknown backend: {config.name!r}")
    return factory(config.kwargs)
