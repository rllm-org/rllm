"""Patches for verl.workers.rollout.vllm_rollout.vllm_async_server module.

This module patches the vLLMAsyncServer class to:
- Add LoRA confg during the creation of the AsyncLLM engine
- Append LoRA request info to the OpenAIServingChat (for /chat/completions endpoint)
- Add LoRA support to the `generate` method directly used for rolling out in AgentExecutionEngine.
"""

from ._utils import LORA_ID_MAGIC, LORA_PATH_MAGIC, get_bounded_args, vllm_max_lora_rank, wrap_class_method_once

TARGET_MODULE = "verl.workers.rollout.vllm_rollout.vllm_async_server"

# ============================================================================
# Patches for vLLMAsyncServer class
# ============================================================================


def init_engine_wrapper(wrapped, instance, args, kwargs):
    """
    Replace init_engine to add LoRA confg during the creation of the AsyncLLM engine.
    """
    import os

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
    from vllm.lora.request import LoRARequest
    from vllm.v1.engine.async_llm import AsyncLLM

    from verl.utils.fs import copy_to_local
    from verl.workers.rollout.vllm_rollout.vllm_async_server import ExternalRayDistributedExecutor, ExternalZeroMQDistributedExecutor

    config = instance.config
    model_path = config.model.path
    model_name = "/".join(model_path.split("/")[-2:])
    local_path = copy_to_local(model_path)
    trust_remote_code = config.model.get("trust_remote_code", False)
    config = config.rollout

    tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
    max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
    max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
    instance.max_model_len = int(max_model_len)

    # Override default generation config from hugging face model config,
    # user can still override them by passing kwargs in each request.
    kwargs = dict(
        n=1,
        logprobs=0,
        repetition_penalty=1.0,
        max_new_tokens=config.response_length,
    )
    for k in config.keys():
        if hasattr(SamplingParams(), str(k)):
            kwargs[k] = config.get(k)
    print(f"override_generation_config: {kwargs}")

    backend = os.environ.get("VERL_VLLM_DISTRIBUTED_BACKEND", "zeromq")
    if backend == "zeromq":
        distributed_executor_backend = ExternalZeroMQDistributedExecutor
    elif backend == "ray":
        distributed_executor_backend = ExternalRayDistributedExecutor
    else:
        distributed_executor_backend = None

    # Fix: we supplement the AsyncEngineArgs with LoRA config.
    instance.lora_rank = instance.config.model.get("lora_rank", 0)

    if instance.lora_rank > 0:
        lora_kwargs = {"enable_lora": True, "max_loras": 1, "max_lora_rank": vllm_max_lora_rank(instance.lora_rank)}
    else:
        lora_kwargs = {}

    engine_args = AsyncEngineArgs(
        model=local_path,
        enable_sleep_mode=config.free_cache_engine,
        override_generation_config=kwargs,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        dtype=config.dtype,
        enforce_eager=config.enforce_eager,
        gpu_memory_utilization=config.gpu_memory_utilization,
        disable_custom_all_reduce=True,
        skip_tokenizer_init=False,
        max_model_len=instance.max_model_len,
        load_format="auto",
        disable_log_stats=config.disable_log_stats,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=config.enable_chunked_prefill,
        enable_prefix_caching=True,
        trust_remote_code=trust_remote_code,
        seed=config.get("seed", 0),
        **lora_kwargs,
    )

    # init async llm engine
    vllm_config = instance._create_engine_config(engine_args)
    instance.engine = AsyncLLM.from_vllm_config(vllm_config)

    # build serving chat
    model_config = instance.engine.model_config
    BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
    models = OpenAIServingModels(instance.engine, model_config, BASE_MODEL_PATHS)
    instance.openai_serving_chat = OpenAIServingChat(
        instance.engine,
        model_config,
        models,
        "assistant",
        request_logger=RequestLogger(max_log_len=4096),
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=config.multi_turn.tool_config_path is not None,
        tool_parser=config.multi_turn.format,  # hermes, llama3_json, ...
    )

    # Fix: we intentionally append the LoRA request in order to support chat completion endpoint.
    if instance.lora_rank > 0:
        lora_request = LoRARequest(lora_name=f"{LORA_ID_MAGIC}", lora_int_id=LORA_ID_MAGIC, lora_path=LORA_PATH_MAGIC)
        instance.openai_serving_chat.models.lora_requests.append(lora_request)


async def chat_completion_wrapper(wrapped, instance, args, kwargs):
    """
    Manually add LoRA model argument to the request so that the request will be routed to the correct LoRA model.
    See https://github.com/volcengine/verl/issues/2048 for discussion.

    API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse

    raw_request: Request = kwargs["raw_request"] if "raw_request" in kwargs else args[0]
    request_json = await raw_request.json()

    if getattr(instance, "lora_rank", 0) > 0:
        request_json["model"] = f"{LORA_ID_MAGIC}"

    request = ChatCompletionRequest(**request_json)
    generator = await instance.openai_serving_chat.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator, media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


# async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> list[int]:
async def generate_wrapper(wrapped, instance, args, kwargs):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.outputs import RequestOutput

    ba = get_bounded_args(wrapped, args, kwargs)
    prompt_ids = ba["prompt_ids"]
    sampling_params = ba["sampling_params"]
    request_id = ba["request_id"]

    max_tokens = instance.max_model_len - len(prompt_ids)
    sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)

    lora_request = LoRARequest(lora_name=f"{LORA_ID_MAGIC}", lora_int_id=LORA_ID_MAGIC, lora_path=LORA_PATH_MAGIC)
    generator = instance.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request)

    # Get final response
    final_res: RequestOutput | None = None
    async for output in generator:
        final_res = output
    assert final_res is not None

    return final_res.outputs[0].token_ids


def register():
    cls_name = "AsyncvLLMServer"
    wrap_class_method_once(TARGET_MODULE, cls_name, "init_engine", init_engine_wrapper)
    wrap_class_method_once(TARGET_MODULE, cls_name, "chat_completion", chat_completion_wrapper)
    wrap_class_method_once(TARGET_MODULE, cls_name, "generate", generate_wrapper)
