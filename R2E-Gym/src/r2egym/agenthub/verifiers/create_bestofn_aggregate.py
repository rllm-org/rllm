import glob
import json
from collections import defaultdict

import fire

from r2egym.agenthub.trajectory.trajectory import Trajectory


def run_ef_verifier(sub_trajs: list[Trajectory]) -> Trajectory:
    """
    Select the trajectory with the highest verifier probability.
    """
    return max(sub_trajs, key=lambda x: x.verifier_prob)


def run_eb_verifier(sub_trajs: list[Trajectory]) -> Trajectory:
    """
    Select the trajectory with the highest execution-based verifier score.
    First, only keep the trajectories with the highest regression score.
    Next, among the trajectories with the highest regression score, select the one with the highest reproduction test score.
    """
    sub_trajs = [
        traj
        for traj in sub_trajs
        if traj.regression_pass_count
        == max(traj.regression_pass_count for traj in sub_trajs)
    ]
    return max(sub_trajs, key=lambda x: x.reproduction_test_score)


def run_hybrid_verifier(sub_trajs: list[Trajectory]) -> Trajectory:
    """
    First, keep top-n trajectories with the highest verifier probability.
    Next, among the top-n trajectories, keep the trajectory with the highest regression score.
    Next, among the trajectories with the highest regression score, keep the ones with the highest reproduction test score.
    Finally, select the trajectory with the highest verifier probability.
    """
    n = len(sub_trajs) // 2
    sub_trajs = sorted(sub_trajs, key=lambda x: x.verifier_prob, reverse=True)[:n]
    sub_trajs = [
        traj
        for traj in sub_trajs
        if traj.regression_pass_count
        == max(traj.regression_pass_count for traj in sub_trajs)
    ]
    sub_trajs = [
        traj
        for traj in sub_trajs
        if traj.reproduction_test_score
        == max(traj.reproduction_test_score for traj in sub_trajs)
    ]
    return max(sub_trajs, key=lambda x: x.verifier_prob)


def run(
    traj_file_glob: str,
    verifier_mode: str,
    output_json_path: str,
):
    assert verifier_mode in ["ef", "eb", "hybrid"]
    verifier_fn_dict = {
        "ef": run_ef_verifier,
        "eb": run_eb_verifier,
        "hybrid": run_hybrid_verifier,
    }
    verifier_fn = verifier_fn_dict[verifier_mode]

    traj_files = glob.glob(traj_file_glob)
    all_trajs_by_docker: dict[str, list[Trajectory]] = defaultdict(list)

    for traj_file in traj_files:
        with open(traj_file, "r") as f:
            for line in f:
                traj = Trajectory.model_validate_json(line)
                all_trajs_by_docker[traj.docker_image].append(traj)

    submission = []
    for docker_image, sub_trajs in all_trajs_by_docker.items():
        selected_traj = verifier_fn(sub_trajs)
        selected_traj.docker_image = docker_image
        submission.append(selected_traj.create_swebench_submission())

    with open(output_json_path, "w") as f:
        json.dump(submission, f)


if __name__ == "__main__":
    fire.Fire(run)
