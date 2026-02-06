import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from rllm.experimental.fully_async.protocol import OutputChunk, OutputWithVersion
from rllm.parser.tool_parser import ToolParser


@dataclass
class ClientStats:
    """Thread-safe HTTP client statistics tracker."""
    in_flight: int = 0
    total_started: int = 0
    total_completed: int = 0
    total_failed: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # Timing stats
    min_latency: float = float('inf')
    max_latency: float = 0.0
    total_latency: float = 0.0
    
    async def start_request(self) -> int:
        async with self._lock:
            self.in_flight += 1
            self.total_started += 1
            return self.in_flight
    
    async def end_request(self, success: bool, latency: float):
        async with self._lock:
            self.in_flight -= 1
            if success:
                self.total_completed += 1
                self.total_latency += latency
                self.min_latency = min(self.min_latency, latency)
                self.max_latency = max(self.max_latency, latency)
            else:
                self.total_failed += 1
    
    async def get_stats(self) -> dict:
        async with self._lock:
            avg_latency = self.total_latency / max(1, self.total_completed)
            return {
                "in_flight": self.in_flight,
                "total_started": self.total_started,
                "total_completed": self.total_completed,
                "total_failed": self.total_failed,
                "avg_latency": avg_latency,
                "min_latency": self.min_latency if self.min_latency != float('inf') else 0,
                "max_latency": self.max_latency,
            }
    
    def get_stats_sync(self) -> dict:
        """Non-async version for quick access (may have slight race)."""
        avg_latency = self.total_latency / max(1, self.total_completed)
        return {
            "in_flight": self.in_flight,
            "total_started": self.total_started,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "avg_latency": avg_latency,
        }


class RolloutClient:
    def __init__(self, router_url: str, tokenizer=None, max_concurrency: int = 4096, max_tokens=32768):
        self.router_url = router_url
        self.tokenizer = tokenizer
        self.parser = ToolParser.get_parser(tokenizer)
        self._max_concurrency = max_concurrency

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self._max_concurrency,
                max_keepalive_connections=min(self._max_concurrency, 1000),
            ),
            timeout=httpx.Timeout(None),
        )

        self.cur_version = 0
        self.max_tokens = max_tokens
        self.resume_event = asyncio.Event()
        self.resume_event.set()
        
        # Request statistics tracking
        self.stats = ClientStats()

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    def set_version(self, version: int):
        self.cur_version = version

    async def _post(self, payload):
        # Block if paused - ensures no new requests after pause()
        await self.resume_event.wait()
        
        start_time = time.time()
        await self.stats.start_request()
        success = False
        try:
            response = await self.client.post(self.router_url + "/generate", json=payload)
            response.raise_for_status()
            success = True
            return response.json()
        finally:
            latency = time.time() - start_time
            await self.stats.end_request(success, latency)

    async def get_stats(self) -> dict:
        """Get current client request statistics."""
        return await self.stats.get_stats()

    def get_stats_sync(self) -> dict:
        """Get stats without awaiting (for quick access in logging)."""
        return self.stats.get_stats_sync()

    def resume(self):
        self.resume_event.set()

    def pause(self):
        self.resume_event.clear()

    # ========== Low-Level API ==========

    async def generate(self, prompt_ids: list[int], sampling_params: dict) -> OutputWithVersion:
        """
        Generate with token IDs directly (low-level API).

        Args:
            prompt_ids: List of input token IDs
            sampling_params: SGLang sampling parameters dict

        Returns:
            OutputWithVersion with prompt_ids and output_chunks
        """
        output = OutputWithVersion(prompt_ids=prompt_ids, output_chunks=[])

        while True:
            # Block at start of each iteration
            await self.resume_event.wait()
            output, sampling_params = await self._generate(output, sampling_params)
            if output.finish_reason == "abort":
                continue
            else:
                return output

    async def _generate(self, output: OutputWithVersion, sampling_params: dict):
        """Internal generate that handles a single request/response cycle."""
        old_version = self.cur_version
        payload = {
            "input_ids": output.all_tokens(),
            "sampling_params": sampling_params,
            "return_logprob": True,
        }

        response = await self._post(payload)

        # finish_reason is a dict with "type" key, or None
        finish_reason_obj = response["meta_info"].get("finish_reason")
        output.finish_reason = finish_reason_obj["type"] if finish_reason_obj else "unknown"

        # output_token_logprobs is a list of tuples: [(log_prob, token_id, _), ...]
        output_token_logprobs = response["meta_info"].get("output_token_logprobs", [])
        # Ensure logprobs are Python floats (not tensors or nested structures)
        logprob_values = [float(log_prob) for log_prob, token_id, _ in output_token_logprobs]

        # TODO: delete this after testing
        output_ids = [token_id for _, token_id, _ in output_token_logprobs]
        assert output_ids == response["output_ids"], "output_ids mismatch, {} != {}".format(output_ids, response["output_ids"])

        chunk = OutputChunk(
            response_ids=response["output_ids"],
            response_logprobs=logprob_values,
            version=old_version if output.finish_reason == "abort" else self.cur_version,
        )
        output.append(chunk)

        # Adjust max_tokens for continuation
        max_tokens = sampling_params.get("max_new_tokens") or sampling_params.get("max_tokens")
        if max_tokens is None:
            return output, sampling_params

        sampling_params = sampling_params.copy()
        remaining = max_tokens - len(chunk.response_ids)
        if "max_new_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = remaining
        else:
            sampling_params["max_tokens"] = remaining

        return output, sampling_params

    # ========== High-Level Chat API ==========
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], OutputWithVersion]:
        """
        Generate chat completion and parse response into OpenAI message format.

        Args:
            messages: List of message dicts (OpenAI format)
            sampling_params: SGLang sampling params dict
            tools: List of tool definitions (OpenAI function calling format)

        Returns:
            (message_dict, output): Parsed message and raw OutputWithVersion
        """
        from rllm.experimental.fully_async.message_utils import parse_response

        if self.tokenizer is None:
            raise ValueError("tokenizer required for chat_completion")

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)

        sampling_params = sampling_params or {}
        if sampling_params.get("max_new_tokens", None) is None:
            sampling_params["max_new_tokens"] = self.max_tokens - len(prompt_ids)

        output = await self.generate(prompt_ids, sampling_params)

        message = parse_response(self.tokenizer, self.parser, output.all_response_ids())
        return message, output

    async def close(self):
        await self.client.aclose()