import json

from fire import Fire

from r2egym.agenthub.trajectory.trajectory import Trajectory


def create_swebench_submission_from_trajectory(trajectory: Trajectory):
    return trajectory.create_swebench_submission()


def create_swebench_submission_from_file(traj_file_path: str, output_json_path: str):
    output = []
    with open(traj_file_path, "r") as f:
        for line in f:
            try:
                trajectory = Trajectory.load_from_model_dump_json(line)
                output.append(create_swebench_submission_from_trajectory(trajectory))
            except Exception as e:
                print(f"Error in create_swebench_submission_from_file: {e}")
                continue
    with open(output_json_path, "w") as f:
        json.dump(output, f)
    return output


if __name__ == "__main__":
    Fire(create_swebench_submission_from_file)
