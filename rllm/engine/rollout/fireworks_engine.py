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

    def update_model_weights(self, lora_adapter_path: dict):
        print("updating fireworks deployment weights")
        pass

    def _upload_model(
        self, model_name, lora_adapter_path: str, base_model: str, account_id: str
    ):
        upload_model_command = f"firectl create model {model_name} {lora_adapter_path} --base-model {base_model} -a {account_id}"
        print(f"running command: {upload_model_command}")
        upload_model_output = os.popen(upload_model_command).readlines()
        print(upload_model_output)
        for line in upload_model_output:
            if line.startswith("Name: "):
                model_name = line.split("Name: ")[-1]
                return model_name.strip()

        raise ValueError(
            f"""
            Error creating model: {upload_model_output}
            Command: {upload_model_command}
            """,
        )
