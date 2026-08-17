from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class GatewaySession:
    session_id: str
    api_key: str


class GatewayManagerProtocol(Protocol):
    def stop(self) -> None: ...

    async def astop(self) -> None: ...

    def create_session(
        self,
        session_id: str,
        sampling_params: dict[str, Any] | None = None,
    ) -> GatewaySession: ...

    async def acreate_session(
        self,
        session_id: str,
        sampling_params: dict[str, Any] | None = None,
    ) -> GatewaySession: ...

    def get_session_url(self, session_id: str, *, public: bool = True) -> str: ...

    async def aget_traces(self, session_id: str) -> list[Any]: ...

    async def adelete_session(self, session_id: str) -> Any: ...
