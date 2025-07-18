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
"""
Generate responses given a dataset of prompts
"""
import csv
import ray
import json
import numpy as np
import hydra
import os
from tabulate import tabulate

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd

from omegaconf import OmegaConf
from transformers import AutoTokenizer
import wandb

from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.protocol import (
    DataProtoItem,
    pad_dataproto_to_divisor,
    unpad_dataproto,
)
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.trainer.ppo.ray_trainer import dataprotoitem_to_dataproto
from verl.utils import hf_tokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.reward_manager import NaiveRewardManager

# Add the select_reward_fn from main_eval.py
def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        from rllm.rewards.rl_reward import rllm_reward_fn
        reward_fn = lambda s, gt: rllm_reward_fn(data_source, s, gt)
        return reward_fn


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    wandb.init(project='verl')
    run_generation(config)


def run_generation(config):

    local_path = copy_to_local(config.model.path)


    tokenizer = hf_tokenizer(local_path)
    if config.rollout.temperature == 0.:
        assert config.rollout.n_val == 1, 'When temperature=0, n_val must be 1.'

    # Load Dataset
    try:
        dataset = pd.read_parquet(config.data.path)
    except Exception as e:
        config.data.path = config.data.path.replace('.parquet', '.json')
        print(f"Error loading dataset: {e}")
        dataset = pd.read_json(config.data.path)



    val_dataset = RLHFDataset(parquet_files=config.data.path,
        tokenizer=tokenizer,
        prompt_key=config.data.prompt_key,
        image_key=config.data.get('image_key', 'images'),
        max_prompt_length=config.rollout.prompt_length,
        filter_prompts=True,
        return_raw_chat=config.data.get('return_raw_chat', False),
        truncation=config.data.get('truncation', 'error'),
        filter_overlong_prompts=config.data.filter_overlong_prompts)
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=len(val_dataset),
        num_workers=8,
        shuffle=False,
        collate_fn=collate_fn)

    # Load Rollout Workers
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='actor_rollout')
    rollout_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    rollout_wg.init_model()

    for test_data in val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)
        n_val = config.rollout.n_val
        test_batch = test_batch.repeat(repeat_times=n_val, interleave=True)

        # Store original inputs
        test_batch.meta_info = {
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': False,
            'validate': True,
        }
        test_batch_padded, pad_size = pad_dataproto_to_divisor(test_batch, rollout_wg.world_size)
        test_output_gen_batch_padded = rollout_wg.generate_sequences(test_batch_padded)
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

    output_ids = test_output_gen_batch_padded.batch['responses']
    output_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    test_batch = dataprotoitem_to_dataproto(test_output_gen_batch) #test_batch.union(test_output_gen_batch)

    # evaluate using reward_function
    val_reward_fn = NaiveRewardManager(tokenizer=tokenizer, num_examine=1)
    # Runs reward calculation on parallel threads.
    reward_tensor = val_reward_fn(test_batch)


    output_lst =  tokenizer.batch_decode(test_batch.batch['input_ids'][:, -config.rollout.response_length:], skip_special_tokens=False)
    output_lst = np.array(output_lst).reshape(len(output_lst)//n_val, n_val).tolist()
    dataset['responses'] = output_lst

    # Save the responses to a new parquet before hand.
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)


    scores = reward_tensor.sum(-1).cpu().tolist()
    scores = np.array(scores).reshape(len(scores)//n_val, n_val).tolist()
    dataset['scores'] = scores
    # # Write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)

    # Max across first dim average correct
    scores = np.array(scores)
    pass_at_n = np.max(scores, axis=1).mean()
    pass_at_1 = np.mean(scores)

    # Save metrics to CSV
    csv_path = os.path.join(output_dir, 'pass.csv')
    results_path = os.path.join(output_dir, 'results.json')

    # Prepare the row data
    # Extract the dataset name from the path
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        f'pass@{n_val}': pass_at_n
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)

    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    with open(results_path, 'w') as f:
        json.dump(scores.tolist(), f)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]

    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

if __name__ == "__main__":
    main()
