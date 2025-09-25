from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any


class StrandsSession:
    """
    Tiny adapter that forwards Strands SDK events as normalized dicts.
    """

    def __init__(self, agent, model_provider):
        self.agent = agent
        self.model_provider = model_provider

    async def step(self, user_msg: str) -> AsyncIterator[dict[str, Any]]:
        async for event in self.agent.stream(user_msg, model=self.model_provider):
            yield event

    async def close(self) -> None:
        if hasattr(self.agent, "aclose"):
            await self.agent.aclose()


async def strands_session_factory(*, system_prompt: str, tools, **kw) -> StrandsSession:
    """
    Factory returning a StrandsSession. Expects `agent` and `model` in kw.
    """
    agent = kw.get("agent")
    model_provider = kw.get("model")
    return StrandsSession(agent, model_provider)
