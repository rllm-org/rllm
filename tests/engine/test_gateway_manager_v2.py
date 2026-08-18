import asyncio
import socket

import httpx
import pytest
from omegaconf import OmegaConf
from rllm_model_gateway.v2 import TokenInput, TokenOutput
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from rllm.gateway.manager import create_gateway_manager
from rllm.gateway.manager_v2 import GatewayManagerV2


class SmokeInferenceClient:
    def __init__(self, completion_token_ids: list[int], weight_version: int) -> None:
        self._completion_token_ids = completion_token_ids
        self._weight_version = weight_version

    async def generate(self, request: TokenInput) -> TokenOutput:
        return TokenOutput(
            completion_token_ids=self._completion_token_ids,
            logprobs=[-0.1] * len(self._completion_token_ids),
            finish_reason="stop",
            weight_version=self._weight_version,
        )

    async def update(self, update: dict) -> None:
        self._weight_version = update["weight_version"]

    async def close(self) -> None:
        pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tokenizer(tmp_path) -> str:
    raw = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "hello": 1, "world": 2, "[EOS]": 3},
            unk_token="[UNK]",
        )
    )
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(tmp_path)
    return str(tmp_path)


def _config(tokenizer_model: str, **overrides):
    gateway = {
        "version": "v2",
        "host": "127.0.0.1",
        "port": _free_port(),
        "num_workers": 1,
        "tokenizer_model": tokenizer_model,
        "renderer_family": "auto",
        "worker_startup_timeout_seconds": 20,
        **overrides,
    }
    return OmegaConf.create({"rllm": {"gateway": gateway}})


def test_factory_constructs_v2_manager_from_gateway_config(tmp_path) -> None:
    config = _config(_tokenizer(tmp_path), num_workers=3, renderer_family="qwen3")

    manager = create_gateway_manager(config, mode="thread")

    assert isinstance(manager, GatewayManagerV2)
    assert manager.host == "127.0.0.1"
    assert manager._gateway_config.num_workers == 3
    assert manager._gateway_config.renderer == "qwen3"
    assert manager._gateway_config.tokenizer_model == str(tmp_path)


def test_v2_requires_a_tokenizer_model() -> None:
    config = OmegaConf.create({"rllm": {"gateway": {"version": "v2"}}})

    with pytest.raises(ValueError, match="tokenizer_model is required"):
        create_gateway_manager(config)


def test_unknown_gateway_version_is_rejected() -> None:
    config = OmegaConf.create({"rllm": {"gateway": {"version": "v3"}}})

    with pytest.raises(ValueError, match="must be 'v1' or 'v2'"):
        create_gateway_manager(config)


def test_v2_manager_head_worker_smoke(tmp_path) -> None:
    manager = create_gateway_manager(_config(_tokenizer(tmp_path)))
    assert isinstance(manager, GatewayManagerV2)

    try:
        manager.start(
            SmokeInferenceClient,
            {"completion_token_ids": [2], "weight_version": 0},
        )
        session = manager.create_session("smoke", {"temperature": 0.5})
        response = httpx.post(
            f"{manager.get_session_url(session.session_id, public=False)}/completions",
            headers={"Authorization": f"Bearer {session.api_key}"},
            json={"model": "model", "prompt": "hello", "max_tokens": 1},
            timeout=10,
        )
        response.raise_for_status()

        assert response.json()["choices"][0]["text"] == "world"
        traces = manager.get_traces("smoke")
        assert len(traces) == 1
        assert traces[0].input.prompt_token_ids == [1]
        assert traces[0].output.completion_token_ids == [2]
        assert traces[0].output.weight_version == 0
        assert traces[0].lineage.parent_request_id is None
        assert traces[0].lineage.root_request_id == traces[0].request.request_id

        asyncio.run(manager.update_inference_client({"weight_version": 2}))
        updated = httpx.post(
            f"{manager.get_session_url(session.session_id, public=False)}/completions",
            headers={"Authorization": f"Bearer {session.api_key}"},
            json={"model": "model", "prompt": "hello", "max_tokens": 1},
            timeout=10,
        )
        updated.raise_for_status()
        assert manager.get_traces("smoke")[-1].output.weight_version == 2
    finally:
        manager.stop()
