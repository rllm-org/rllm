from typing import Literal, Optional

import torch
from vllm.inputs import SingletonInputs
from vllm.lora.request import LoRARequest
from vllm.multimodal.processing import EncDecMultiModalProcessor

#  vLLM 0.7.X
def sleep_v1(self, level: int = 1):
    self.engine.reset_prefix_cache()
    self.engine.sleep(level=level)

def wake_up_v1(self):
    self.engine.wake_up()

# vLLM 0.8.X implements sleep and wake_up as a coroutine, so we implement a synchronous version.
def sleep_v2(self, level: int = 1) -> None:
    self.engine.reset_prefix_cache()
    self.engine.sleep(level)

def wake_up_v2(self, tags: Optional[list[str]] = None) -> None:
    self.engine.wake_up(tags)

def _validate_model_input(
        self,
        prompt_inputs: SingletonInputs,
        lora_request: Optional[LoRARequest],
        *,
        prompt_type: Literal["encoder", "decoder"],
    ):
    model_config = self.model_config
    tokenizer = (None if self.tokenizer is None else
                    self.tokenizer.get_lora_tokenizer(lora_request))

    prompt_ids = prompt_inputs["prompt_token_ids"]
    if prompt_ids is None or len(prompt_ids) == 0:
        if prompt_type == "encoder" and model_config.is_multimodal_model:
            pass  # Mllama may have empty encoder inputs for text-only data
        else:
            raise ValueError(f"The {prompt_type} prompt cannot be empty")

    max_prompt_len = self.model_config.max_model_len
    if len(prompt_ids) > max_prompt_len:
        if prompt_type == "encoder" and model_config.is_multimodal_model:
            mm_registry = self.input_preprocessor.mm_registry
            mm_processor = mm_registry.create_processor(
                model_config,
                tokenizer=tokenizer or object(),  # Dummy if no tokenizer
            )
            assert isinstance(mm_processor, EncDecMultiModalProcessor)

            if mm_processor.pad_dummy_encoder_prompt:
                return  # Skip encoder length check for Whisper

        if model_config.is_multimodal_model:
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens plus multimodal tokens. For image "
                "inputs, the number of image tokens depends on the number "
                "of images, and possibly their aspect ratios as well.")
        else:
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens.")

        raise ValueError(
            f"The {prompt_type} prompt (length {len(prompt_ids)}) is "
            f"longer than the maximum model length of {max_prompt_len}. "
            f"{suggestion}")

        # TODO: Find out how many placeholder tokens are there so we can
        # check that chunked prefill does not truncate them
        # max_batch_len = self.scheduler_config.max_num_batched_tokens



def all_reduce(self, input_):
    ca_comm = self.ca_comm
    if ca_comm is not None and not ca_comm.disabled and \
        ca_comm.should_custom_ar(input_):
        out = ca_comm.custom_all_reduce(input_)
        assert out is not None
        return out
    from vllm.utils import current_stream
    stream = current_stream()
    out = input_.clone()
    with torch.cuda.stream(stream):
        torch.distributed.all_reduce(out, group=self.device_group)
    return out
