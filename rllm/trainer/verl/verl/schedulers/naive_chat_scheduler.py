# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
from copy import deepcopy
from typing import Any, Dict, List, Union
import numpy as np
import torch
import os
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict
from jinja2 import Environment, FileSystemLoader, select_autoescape

from verl.protocol import DataProto
from verl.workers.rollout.async_server import ChatCompletionScheduler

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

def patch_tokenizer_chat_template_fn(tokenizer):
    if '<think>' in tokenizer.get_vocab() and tokenizer.eos_token == "<｜end▁of▁sentence｜>":
            # Create a Jinja Template object only once
        template_dir = os.path.dirname(__file__)
        env = Environment(
            loader=FileSystemLoader(searchpath=template_dir),  # Use current directory as root
            autoescape=select_autoescape(enabled_extensions=[]),
        )
        template = env.get_template('deepseek_qwen.jinja')
        def _apply_deepseek_chat_template(self, messages, **kwargs):
            # Gather any keyword arguments that the template expects.
            # In this case, `add_generation_prompt` defaults to False if not passed.
            context = {
                "messages": messages,
                "add_generation_prompt": kwargs.get("add_generation_prompt", False),
                "tokenize": kwargs.get("tokenize", False),
            }
            # Render the template to a single string
            rendered = template.render(**context)
            return rendered
        tokenizer.apply_chat_template = _apply_deepseek_chat_template.__get__(tokenizer, type(tokenizer))

class NaiveChatCompletionScheduler(ChatCompletionScheduler):
    """
    A very naive implementation of ChatCompletionScheduler for demo purpose,
    only do single-turn chat completion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        patch_tokenizer_chat_template_fn(self.tokenizer)

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            #logprobs=1,
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
        print(f"[NaiveChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            assert exception is None, f"exception: {exception}"
            conversation, batch_conversations, batch_index = (
                info["conversation"],
                info["batch_conversations"],
                info["batch_index"],
            )

            # comps = []
            # for choice in completions.choices:
            #     token_ids= choice.logprobs.content
            #     token_ids = [int(t.token.split(":")[1]) for t in token_ids]
            #     comps.append(token_ids)
            # batch_conversations[batch_index] = comps


            conversations = []
            for choice in completions.choices:
                chat = conversation.copy()
                chat.append({"role": choice.message.role, "content": choice.message.content})
                conversations.append(chat)
            batch_conversations[batch_index] = conversations

            # NOTE: we can call tools and resubmit chat completions here.
            # call_tools(completions, info)
            # await self.submit_chat_completions(callback2, ...)

        # TODO: we may need to control max concurrent requests here, or it will harm prefix cache hit rate.
        tasks, batch_conversations = [], [None] * len(batch)
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "conversation": list(conversation),
                        },
                        model=self.model_name,
                        messages=conversation.tolist(),
                        **kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)
        print("[NaiveChatCompletionScheduler] generate_sequences done")

        return self._postprocess(batch, batch_conversations, kwargs["n"])

    def _postprocess(self, batch: DataProto, batch_conversations: List[List[List[Dict[str, str]]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch["raw_prompt"]]
        non_tensor_batch = deepcopy(batch.non_tensor_batch)
        meta_info = deepcopy(batch.meta_info)
        # flatten batch_conversations if n > 1
        assert len(batch_conversations) == len(prompts)
        batch_conversations = [conversation for conversations in batch_conversations for conversation in conversations]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False) for conversation in batch_conversations]

        # responses: [response]
        # TODO: mask out tools calling tokens?
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding='max_length', max_length=self.config.prompt_length, padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding='max_length', truncation=True,max_length=self.config.response_length, padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)
            for key, val in non_tensor_batch.items():
                non_tensor_batch[key] = _repeat_interleave(val, n)


        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        final_batch = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )
        return DataProto(batch=final_batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
