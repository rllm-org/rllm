import asyncio
import logging
import os

import openai

from rllm.engine.rollout.openai_engine import OpenAIEngine
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START
from rllm.parser import ChatTemplateParser, ToolParser


class FireworksEngine(OpenAIEngine):
    def __init__(
        self,
        model: str,
        tokenizer=None,
        api_retries: int = 3,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        api_key: str = os.getenv("FIREWORKS_API_KEY"),
        sampling_params: dict | None = None,
        **kwargs,
    ):
        self.model = model
        self.api_retries = api_retries
        self.sampling_params = sampling_params or {}
        self._use_chat_completions = True
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))
            try:
                self.tool_parser = ToolParser.get_parser(self.tokenizer)
            except Exception:
                print(f"Warning: No tool parser found for {self.tokenizer.name_or_path}. Tool calls not be parsed.")
                self.tool_parser = None
            self._use_chat_completions = False
        else:
            print("No tokenizer provided, will use the chat completions endpoint. This is not recommended.")
            self._use_chat_completions = True

        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def update_model_weights(self, state_dict: dict):
        pass
