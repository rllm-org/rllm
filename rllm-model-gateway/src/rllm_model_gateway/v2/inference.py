from typing import Any, Protocol

from rllm_model_gateway.v2.types import TokenInput, TokenOutput


class InferenceClient(Protocol):
    async def generate(self, request: TokenInput) -> TokenOutput: ...

    async def update(self, update: dict[str, Any]) -> None: ...

    async def close(self) -> None: ...


InferenceClientClass = type[InferenceClient]
