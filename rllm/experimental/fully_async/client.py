import asyncio
from typing import Any, Literal

import httpx

from rllm.experimental.fully_async.protocol import OutputChunk, OutputWithVersion


class RolloutClient:
    def __init__(self, router_url: str, tokenizer=None, max_concurrency: int = 4096):
        self.router_url = router_url
        self.tokenizer = tokenizer
        self._max_concurrency = max_concurrency

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self._max_concurrency,
                max_keepalive_connections=min(self._max_concurrency, 1000),
            ),
            timeout=httpx.Timeout(None),
        )

        self.cur_version = 0
        self.aborted_queue = asyncio.Queue()
        self.resume_event = asyncio.Event()
        self.resume_event.set()

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    def set_version(self, version: int):
        self.cur_version = version

    async def _post(self, payload):
        response = await self.client.post(self.router_url + "/generate", json=payload)
        response.raise_for_status()
        return response.json()

    def resume(self):
        self.resume_event.set()

    def pause(self):
        self.resume_event.clear()

    # ========== OpenAI-Compatible API ==========

    async def generate_chat(
        self,
        # === OpenAI Chat Completions API parameters ===
        messages: list[dict[str, Any]],
        model: str | None = None,  # ignored, for API compatibility
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stop: str | list[str] | None = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool = True,  # Default True for training
        top_logprobs: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        response_format: dict | None = None,
        seed: int | None = None,
        user: str | None = None,
        # === SGLang extra parameters ===
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        stop_token_ids: list[int] | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        chat_template_kwargs: dict | None = None,
        continue_final_message: bool = False,
    ) -> OutputWithVersion:
        """
        Generate completion with OpenAI-compatible API.
        Matches /v1/chat/completions parameter names exactly.

        Returns:
            OutputWithVersion with prompt_ids and output_chunks containing
            response token IDs and log probabilities.
        """
        if self.tokenizer is None:
            raise ValueError("tokenizer is required for generate_chat(). Pass tokenizer to RolloutClient.__init__() or use generate() with input_ids.")

        # Step 1: Apply chat template to get prompt_ids
        prompt_ids = self._apply_chat_template(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
            continue_final_message=continue_final_message,
        )

        # Step 2: Build sampling params for SGLang /generate
        sampling_params = self._build_sampling_params(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_completion_tokens or max_tokens,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            n=n,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            stop_token_ids=stop_token_ids,
            seed=seed,
            logit_bias=logit_bias,
        )

        # Step 3: Generate with auto-resume using existing logic
        return await self.generate(prompt_ids, sampling_params)

    def _apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        reasoning_effort: str | None = None,
        chat_template_kwargs: dict | None = None,
        continue_final_message: bool = False,
    ) -> list[int]:
        """
        Apply chat template exactly like /v1/chat/completions.

        This matches the behavior of SGLang's serving_chat.py:
        - Applies tokenizer.apply_chat_template with add_generation_prompt=True
        - Handles tools by converting to function format
        - Supports continue_final_message for continuing assistant responses
        """
        # Convert tools to function format (matching serving_chat.py)
        processed_tools = None
        if tools and tool_choice != "none":
            processed_tools = []
            for t in tools:
                if "function" in t:
                    processed_tools.append(t["function"])
                else:
                    processed_tools.append(t)

        # Handle continue_final_message
        working_messages = [msg.copy() for msg in messages]
        assistant_prefix = None

        if continue_final_message and working_messages and working_messages[-1].get("role") == "assistant":
            assistant_prefix = working_messages[-1].get("content", "")
            working_messages = working_messages[:-1]

        # Build template kwargs
        template_kwargs = {}
        if processed_tools:
            template_kwargs["tools"] = processed_tools
        if reasoning_effort:
            template_kwargs["reasoning_effort"] = reasoning_effort
        if chat_template_kwargs:
            template_kwargs.update(chat_template_kwargs)

        # Apply template
        prompt_ids = self.tokenizer.apply_chat_template(
            working_messages,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )

        # Ensure prompt_ids is a list
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)

        # Append assistant prefix if continuing
        if assistant_prefix:
            encoded = self.tokenizer.encode(assistant_prefix, add_special_tokens=False)
            # Remove BOS if present
            if encoded and hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
                if encoded[0] == self.tokenizer.bos_token_id:
                    encoded = encoded[1:]
            prompt_ids = prompt_ids + list(encoded)

        return prompt_ids

    def _build_sampling_params(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        stop: str | list[str] | None = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        n: int = 1,
        top_k: int | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Build sampling params dict for SGLang /generate endpoint."""
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "n": n,
        }

        # Use max_new_tokens for SGLang
        if max_tokens is not None:
            params["max_new_tokens"] = max_tokens

        if stop is not None:
            params["stop"] = stop
        if top_k is not None:
            params["top_k"] = top_k
        if min_p is not None:
            params["min_p"] = min_p
        if repetition_penalty is not None:
            params["repetition_penalty"] = repetition_penalty
        if stop_token_ids is not None:
            params["stop_token_ids"] = stop_token_ids
        if seed is not None:
            params["sampling_seed"] = seed
        if logit_bias is not None:
            params["logit_bias"] = logit_bias

        return params

    # ========== Original Low-Level API (preserved for backward compatibility) ==========

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
        version = self.cur_version
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
        logprob_values = [log_prob for log_prob, token_id, _ in output_token_logprobs]

        chunk = OutputChunk(
            response_ids=response["output_ids"],
            response_logprobs=logprob_values,
            version=version,
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

    async def close(self):
        await self.client.aclose()
