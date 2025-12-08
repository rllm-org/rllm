import os
import fire
import glob
import json
import math
import traceback
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset, Dataset, concatenate_datasets
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
import multiprocessing as mp
import time

import numpy as np
from tqdm import tqdm

from r2egym.logging import setup_logging, INFO
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent
from r2egym.agenthub.trajectory.trajectory import Trajectory

agent_args = AgentArgs.from_yaml(
    Path("./src/r2egym/agenthub/config/r2egym/edit_fn_calling.yaml")
)

logger = setup_logging(
    name="testrollouteval", level=INFO, log_file="testrollouteval.log", console=True
)


def run_test_patch(ds, test_patch, patch):
    """
    Applies a patch in the given environment, runs a test command, and computes the predicted reward.

    If the output contains the word 'resolved', the predicted reward is 1.0, otherwise 0.0.
    The patch is undone after testing.
    """
    # print(ds)
    name = ds["docker_image"].replace("/", "_") + str(hash(patch + test_patch))
    custom_logger = setup_logging(
        name=name,
        log_file=f"run_logs/run_reproduction_tests/{name}.log",
        console=True,
        level=INFO,
    )
    custom_logger.info(f"Starting run_test_patch for {name}!!")
    try:
        name = ds["docker_image"].replace("/", "_") + str(hash(patch + test_patch))
        env_args = EnvArgs(ds, docker_image=ds["docker_image"])
        env = RepoEnv(env_args, logger=custom_logger)  # , backend="kubernetes")
        env.add_commands(agent_args.command_files)
        if test_patch:
            out, err = env.runtime.apply_patch(test_patch)
            if err != "0":
                custom_logger.error(
                    f"WARNING WARNING some error is test application: {out}"
                )
        else:
            return 0
        try:
            # Apply the patch if provided
            if patch:  # -> check for empty patches
                out, err = env.runtime.apply_patch(patch)
                if err != "0":
                    custom_logger.error(
                        f"WARNING WARNING some error is application: {out}"
                    )
            # Execute the test command
            custom_logger.info(f"Executing test command:")
            out, error_code = env.runtime.run(
                "execute_bash --cmd 'python3 test_issue.py -v'"
            )
            custom_logger.info(f"Test output:\n{out}")

            pred_reward = out.count("resolved")
            custom_logger.info(f"Predicted reward: {pred_reward}")

        except Exception as e:
            custom_logger.error(f"Error during patch testing: {e}")
            custom_logger.error(traceback.format_exc())
            pred_reward = 0.0
        return pred_reward
    except Exception as e:
        logger.error(f"Error in Docker runtime: {e}")
        logger.error(traceback.format_exc())
        return 0.0
    finally:
        env.close()


def process_single_task(args, timeout=600, max_retries=3):
    ds, test_patch, patch, test_index = args

    for attempt in range(max_retries):
        try:
            result_queue = mp.Queue()

            def worker():
                try:
                    result = run_test_patch(ds, test_patch, patch)
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))

            process = mp.Process(target=worker)
            process.start()
            process.join(timeout=timeout)

            if process.is_alive():
                # Process timed out, terminate it
                print(
                    f"Attempt {attempt + 1}/{max_retries} timed out for {ds['docker_image']}, terminating process..."
                )
                process.terminate()
                process.join(timeout=5)  # Give it 5 seconds to clean up
                if process.is_alive():
                    process.kill()  # Force kill if still alive
                    process.join()
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # exponential backoff
                continue

            # Process completed, get result
            if not result_queue.empty():
                status, result = result_queue.get()
                if status == "success":
                    return ds["docker_image"], test_index, result
                else:
                    raise Exception(result)
            else:
                raise Exception("Process completed but no result returned")

        except Exception as e:
            print(
                f"Attempt {attempt + 1}/{max_retries} failed for {ds['docker_image']}: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # exponential backoff

    # All retries failed, return 0 reward
    print(f"All {max_retries} attempts failed for {ds['docker_image']}")
    return ds["docker_image"], test_index, 0.0


def load_reproduction_tests():
    ds = (
        load_dataset("R2E-Gym/R2E-TestgenAgent-Patches", split="train")
        .to_pandas()
        .set_index("docker_image")
    )
    return ds


def add_reproduction_tests(trajectories: list[Trajectory], max_workers: int = 32):
    reproduction_tests_df = load_reproduction_tests()

    all_tasks = []

    for trajectory in trajectories:
        docker_image = trajectory.docker_image
        if docker_image in reproduction_tests_df.index:
            reproduction_tests = json.loads(
                reproduction_tests_df.loc[docker_image, "test_patches"]
            )
            for test_idx, test_patch in enumerate(reproduction_tests):
                all_tasks.append(
                    (trajectory.ds, test_patch, trajectory.true_output_patch, test_idx)
                )

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(
            tqdm(ex.map(process_single_task, all_tasks), total=len(all_tasks))
        )

    # defaultdict(defaultdict(int))
    results_dict = defaultdict(lambda: defaultdict(int))
    for docker_image, test_idx, result in results:
        results_dict[docker_image][test_idx] = result

    for trajectory in trajectories:
        results = results_dict[trajectory.docker_image]
        sorted_values_by_test_idx = [
            x[1] for x in sorted(results.items(), key=lambda x: x[0])
        ]
        trajectory.reproduction_test_scores = sorted_values_by_test_idx

    return trajectories
