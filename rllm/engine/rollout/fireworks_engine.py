import asyncio
import json
import os
import time
from urllib.parse import urljoin

import openai
import requests
from fireworks.control_plane.generated.protos_grpcio.gateway.deployed_model_pb2 import (
    DeployedModel as SyncDeployedModel,
)
from fireworks.control_plane.generated.protos_grpcio.gateway.deployed_model_pb2 import (
    ListDeployedModelsRequest as SyncListDeployedModelsRequest,
)
from fireworks.gateway import Gateway

from rllm.engine.rollout.openai_engine import OpenAIEngine
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.globals import THOUGHT_DELIMITER_END, THOUGHT_DELIMITER_START


class FireworksEngine(OpenAIEngine):
    def __init__(
        self,
        deployment_id: str,
        tokenizer=None,
        api_retries: int = 3,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        api_key: str = os.getenv("FIREWORKS_API_KEY"),
        sampling_params: dict | None = None,
        **kwargs,
    ):
        gateway = Gateway()
        self._account_id = gateway.account_id()
        self._deployment_id = deployment_id

        formatted_deployment_id = f"accounts/{self._account_id}/deployments/{deployment_id}"
        deployment = gateway.get_deployment_sync(formatted_deployment_id)
        self._base_model = deployment.base_model

        list_model_request = SyncListDeployedModelsRequest(filter=f'deployment="{formatted_deployment_id}"')
        list_model_response = gateway.list_deployed_models_sync(list_model_request)
        assert list_model_response.total_size == 1, f"Expected only one model under deployment {formatted_deployment_id}"
        deployed_model = list_model_response.deployed_models[0]
        model_name = deployed_model.name
        assert deployed_model.state == SyncDeployedModel.DEPLOYED, f"Expected {model_name} in state DEPLOYED"

        super().__init__(
            model=model_name,
            tokenizer=tokenizer,
            api_retries=api_retries,
            base_url=base_url,
            api_key=api_key,
            sampling_params=sampling_params,
            **kwargs,
        )
        self._use_chat_completions = True  # Always True for Fireworks

    def update_model_weights(self, fireworks_model_id: str, lora_adapter_path: dict) -> bool:
        self._upload_lora(fireworks_model_id, lora_adapter_path, self._base_model, self._account_id)
        self._hot_load_lora(fireworks_model_id, self._deployment_id, self._account_id)

        self.model = f"{self._account_id}/{fireworks_model_id}#{self._account_id}/{self._deployment_id}"
        is_deployment_ready = asyncio.run(self._probe_deployment(self.model))
        return is_deployment_ready

    def _upload_lora(self, fireworks_model_id, lora_adapter_path: str, base_model: str, account_id: str) -> None:
        upload_model_command = f"firectl create model {fireworks_model_id} {lora_adapter_path} --base-model {base_model} -a {account_id} --output json"
        print(f"running command: {upload_model_command}")
        upload_model_output = os.popen(upload_model_command).read()
        print("Fireworks upload model message: ", upload_model_output)
        upload_model_output = json.loads(upload_model_output)

        assert fireworks_model_id in upload_model_output.get("name")
        assert upload_model_output["state"].lower() == "ready"
        print(f"Successfully uploaded model {fireworks_model_id}")

    def _hot_load_lora(self, model_id: str, deployment: str, account_id: str) -> None:
        load_lora_command = f"firectl load-lora {model_id} --deployment {deployment} --replace-merged-addon -a {account_id}"
        print(f"Running command: {load_lora_command}")
        load_lora_output = os.popen(load_lora_command).read()
        print(load_lora_output)

    async def _probe_deployment(self, model_name) -> bool:
        print("Probing model: ", model_name)
        while True:
            try:
                _ = await self.client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": "hi"}])

                # TODO(tianyi): Remove after landing https://github.com/BerriAI/litellm/pull/15938/
                gateway = Gateway()
                formatted_deployment_id = f"accounts/{self._account_id}/deployments/{self._deployment_id}"
                list_model_request = SyncListDeployedModelsRequest(filter=f'deployment="{formatted_deployment_id}"')
                list_model_response = gateway.list_deployed_models_sync(list_model_request)
                assert list_model_response.total_size == 1, f"Expected only one model under deployment {formatted_deployment_id}"
                deployed_model = list_model_response.deployed_models[0]
                self.model = deployed_model.name

                return True
            except Exception as e:
                error_message = str(e).lower()
                print(error_message)
                if "404" in error_message:
                    time.sleep(10)
                    continue
                if "502" in error_message:
                    time.sleep(10)
                    continue
                else:
                    return False

    async def chat_completion(self, messages: list[dict], **kwargs) -> ModelOutput:
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)
        kwargs.pop("model", None)
        kwargs.pop("enforce_max_prompt_length", None)

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        create_params = self._prepare_max_tokens_param(sampling_params)

        retries = self.api_retries
        while retries > 0:
            try:
                merged_sampling_params = {**create_params, **sampling_params}
                response = self._fireworks_chat_completion(messages=messages, sampling_params=merged_sampling_params)
                content = response["choices"][0]["message"]["content"]
                reasoning = response["choices"][0]["message"].get("reasoning", "")
                tool_calls = response["choices"][0]["message"].get("tool_calls", [])

                # Build text with reasoning if available, otherwise use content
                if reasoning:
                    text = f"{THOUGHT_DELIMITER_START}\n{reasoning}\n{THOUGHT_DELIMITER_END}\n\n{content}"
                else:
                    text = content

                prompt_length = response["usage"]["prompt_tokens"]
                completion_length = response["usage"]["completion_tokens"]
                finish_reason = response["choices"][0]["finish_reason"]

                prompt_token_ids = response["prompt_token_ids"]
                completion_token_ids = response.json()["choices"][0]["token_ids"]
                return ModelOutput(
                    text=text,
                    content=content,
                    reasoning=reasoning,
                    tool_calls=tool_calls,
                    prompt_ids=prompt_token_ids,
                    completion_ids=completion_token_ids,
                    prompt_length=prompt_length,
                    completion_length=completion_length,
                    finish_reason=finish_reason,
                )

            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    raise Exception("Rate limit reached and retries exhausted.") from None
                print("Sleep for 5 seconds for API limit.")
                await asyncio.sleep(5)

            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise Exception(f"Error processing content after retries: {e}") from e
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)

    def _fireworks_chat_completion(self, messages, sampling_params):
        url = urljoin(str(self.client.base_url), "/chat/completions")
        payload = {
            "model": self.model,
            "messages": messages,
            **sampling_params,
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.client.api_key}",
        }
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        return response.json()
