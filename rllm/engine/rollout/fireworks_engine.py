import asyncio
import json
import os
import time

from fireworks.control_plane.generated.protos_grpcio.gateway.deployed_model_pb2 import (
    DeployedModel as SyncDeployedModel,
)
from fireworks.control_plane.generated.protos_grpcio.gateway.deployed_model_pb2 import (
    ListDeployedModelsRequest as SyncListDeployedModelsRequest,
)
from fireworks.gateway import Gateway

from rllm.engine.rollout.openai_engine import OpenAIEngine


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

    def update_model_weights(self, fireworks_model_id: str, lora_adapter_path: dict):
        self._upload_lora(fireworks_model_id, lora_adapter_path, self._base_model, self._account_id)
        self._hot_load_lora(fireworks_model_id, self._deployment_id, self._account_id)

        self.model = f"{self._account_id}/{fireworks_model_id}#{self._account_id}/{self._deployment_id}"
        asyncio.run(self._probe_deployment(self.model))

    def _upload_lora(self, fireworks_model_id, lora_adapter_path: str, base_model: str, account_id: str):
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

    async def _probe_deployment(self, model_name) -> None:
        print("Probing model: ", model_name)
        while True:
            try:
                _ = await self.client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": "hi"}])
                return
            except Exception as e:
                error_message = str(e).lower()
                if "404" in error_message:
                    time.sleep(10)
                    continue
                if "502" in error_message:
                    time.sleep(10)
                    print(error_message)
                    continue
                else:
                    raise ValueError(e)
