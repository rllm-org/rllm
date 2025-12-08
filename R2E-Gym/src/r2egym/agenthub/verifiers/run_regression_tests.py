from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import fire
import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset

from r2egym.logging import setup_logging
from r2egym.agenthub.agent.agent import AgentArgs
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.trajectory.trajectory import Trajectory

agent_args = AgentArgs.from_yaml(
    Path("./src/r2egym/agenthub/config/r2egym/edit_fn_calling.yaml")
)
swebv_dataset = (
    load_dataset("r2e-edits/swebench-verified-v2", split="test")
    .to_pandas()
    .set_index("docker_image")
)


def compute_regression_output(trajectory: Trajectory, mode="modeloutput"):
    docker_image = trajectory.docker_image
    run_tests_regression = swebv_dataset.loc[docker_image, "run_tests_regression"]

    ds = trajectory.ds
    env_args = EnvArgs(ds=ds)
    logger = setup_logging(f"REGRESSION_{docker_image}", console=False)
    env = RepoEnv(env_args, logger=logger)
    env.reset()
    env.add_commands(agent_args.command_files)

    if mode == "modeloutput":
        if trajectory.true_output_patch:
            msg, succ = env.runtime.apply_patch(trajectory.true_output_patch)
            if succ != "0":
                print(
                    "FAILED PATCH -- ",
                    msg.strip(),
                    trajectory.reward,
                    trajectory.docker_image,
                )
    elif mode == "gt":
        msg, succ = env.runtime.apply_patch(trajectory.ds["patch"])
        if succ != "0":
            print(
                "FAILED PATCH -- ",
                msg.strip(),
                trajectory.reward,
                trajectory.docker_image,
            )
    elif mode == "no":
        pass

    output = env.runtime.run_swebv_regression(run_tests_regression)
    env.close()
    return output


def add_regression_output(trajectories: list[Trajectory], max_workers: int = 42):

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        reg_outputs = list(
            tqdm.tqdm(
                executor.map(compute_regression_output, trajectories),
                total=len(trajectories),
            )
        )

    for trajectory, reg_output in zip(trajectories, reg_outputs):
        trajectory.regression_test_output = reg_output

    return trajectories
