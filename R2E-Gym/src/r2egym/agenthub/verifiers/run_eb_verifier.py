import glob

import fire

from r2egym.agenthub.trajectory.trajectory import Trajectory
from r2egym.agenthub.verifiers.run_regression_tests import add_regression_output
from r2egym.agenthub.verifiers.run_reproduction_tests import add_reproduction_tests


def process_trajectories_to_verifier_format(
    traj_file_glob: str,
    max_workers: int = 42,
):
    traj_files = glob.glob(traj_file_glob)
    for traj_file in traj_files:
        trajectories: list[Trajectory] = []
        with open(traj_file, "r") as f:
            for line in f:
                trajectories.append(Trajectory.model_validate_json(line))

        trajectories = trajectories
        trajectories = add_regression_output(trajectories, max_workers=max_workers)
        trajectories = add_reproduction_tests(trajectories, max_workers=max_workers)

        with open(traj_file, "w") as f:
            for traj in trajectories:
                f.write(traj.model_dump_json() + "\n")


if __name__ == "__main__":
    fire.Fire(process_trajectories_to_verifier_format)
