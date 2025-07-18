import asyncio
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch
from omegaconf import DictConfig
from openai.types.completion import Completion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.async_server import ChatCompletionScheduler

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

class CompletionsScheduler(ChatCompletionScheduler):
    """
    Implementation of CompletionsScheduler for OpenAI's Completion API.
    """
    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        """
        Args:
            config: DictConfig, rollout config.
            model_path: str, model path.
            server_addresses: List[str], server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        super().__init__(config, model_path, server_addresses, max_cache_size)

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id


    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_tokens=self.config.response_length,  # Changed from max_completion_tokens
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            logprobs=1,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        if is_validate:
            kwargs.update({
                #'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            })

        if batch.meta_info.get("max_tokens", None) is not None:
            kwargs['max_tokens'] = batch.meta_info['max_tokens']

        if batch.meta_info.get('agent_rollout', False):
            kwargs['n'] = 1

        kwargs.update(sampling_params)
        #print(f"[OpenAICompletionScheduler] generate_sequences sampling params: {kwargs}")

        async def callback(completions: Completion, info: Dict[str, Any], exception: Exception):
            batch_response_ids, batch_index = (
                info["batch_response_ids"],
                info["batch_index"],
            )

            # Handle Completion response format which has 'choices' with 'text' instead of 'message'
            comps = []
            for choice in completions.choices:
                token_ids= choice.logprobs.tokens
                token_ids = [int(t.split(":")[1]) for t in token_ids]
                comps.append(token_ids)
            batch_response_ids[batch_index] = comps

        # TODO: we may need to control max concurrent requests here, or it will harm prefix cache hit rate.
        tasks, batch_response_ids = [], [None] * len(batch)
        for batch_index, formatted_prompt in enumerate(batch.non_tensor_batch["formatted_prompts"]):
            # For Completion API, we need to convert the conversation to a prompt string

            tasks.append(
                asyncio.create_task(
                    self.submit_completions(  # Changed from submit_chat_completions
                        callback=callback,
                        callback_additional_info={
                            "batch_response_ids": batch_response_ids,
                            "batch_index": batch_index,
                        },
                        model=self.model_name,
                        prompt=formatted_prompt,  # Changed from messages
                        **kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)
        #print("[OpenAICompletionScheduler] generate_sequences done")
        return self._postprocess(batch, batch_response_ids, kwargs["n"])

    def _postprocess(self, batch: DataProto, response_ids: List[List[str]], n: int) -> DataProto:
        # NOTE: For Completion API, batch_completions is a list of lists of strings (not dictionaries)
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        idx = batch.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = batch.batch["attention_mask"]
        position_ids = batch.batch["position_ids"]
        non_tensor_batch = deepcopy(batch.non_tensor_batch)

        # Flatten to list.
        # Flatten the list of lists of token IDs
        response = []
        for r_ids in response_ids:
            if r_ids is not None:  # Ensure we don't process None values
                for r in r_ids:
                    response.append(r)
        assert len(response) == len(non_tensor_batch["formatted_prompts"]) * n
        response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)

        if n > 1:
            idx = _repeat_interleave(idx, n)
            attention_mask = _repeat_interleave(attention_mask, n)
            position_ids = _repeat_interleave(position_ids, n)
            for key, val in non_tensor_batch.items():
                non_tensor_batch[key] = _repeat_interleave(val, n)

        batch_size = len(idx)
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=self.eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        output = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=output, meta_info=batch.meta_info)
