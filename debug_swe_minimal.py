#!/usr/bin/env python3
import os
from datasets import load_dataset
from rllm.environments.swe.swe import SWEEnv

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DOCKER_MIRROR_PREFIX", "aibrix-docker-mirror-cn-beijing.cr.volces.com")

dataset_name = os.getenv("DATASET_NAME", "R2E-Gym/R2E-Gym-Subset")
instance_idx = int(os.getenv("INSTANCE_IDX", "1000"))

ds = load_dataset(dataset_name, split="train")
entry = ds[instance_idx]

env = SWEEnv(
    entry=entry,
    backend='kubernetes',
    scaffold='r2egym',
    step_timeout=120,
    reward_timeout=300,
    delete_image=False,
    verbose=True,
)

task_instruction, info = env.reset()

obs, reward, done, info = env.step("execute_bash pwd")

env.close()
