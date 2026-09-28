import os, torch
from vllm import LLM, SamplingParams
print("torch", torch.__version__, "| devices", torch.cuda.device_count())
llm = LLM(model="Qwen/Qwen3-4B-Instruct-2507", max_model_len=2048,
          gpu_memory_utilization=0.35, enforce_eager=True, tensor_parallel_size=1)
out = llm.generate(["What is 137 * 24? Answer with the number only."],
                   SamplingParams(temperature=0.0, max_tokens=48))
print("=== GENERATION ===")
print(repr(out[0].outputs[0].text))
