import sys
import numpy as np
import pandas as pd
import json  # for potential debugging or alternative parsing
import fire  # for command-line interface
from r2egym.agenthub.action import Action
from r2egym.agenthub.trajectory.trajectory import TrajectoryStep, Trajectory

def analyze_log(
    filename: str, remove_traj_time_limit: bool = False, minimal: bool = False
):
    """
    Processes a JSONL file containing trajectory dumps.

    Args:
        filename (str): The path to the JSONL file.
        remove_traj_time_limit (bool): If True, prompt to remove all trajectories whose exit_reason
            equals "traj_time_limit" from the file.
    """
    if remove_traj_time_limit:
        # Load the file lines.
        with open(filename, "r") as f:
            lines = f.readlines()

        # Identify lines to remove.
        new_lines = []
        removal_count = 0
        for line in lines:
            # Load each trajectory using the provided helper.
            t = Trajectory.load_from_model_dump_json(line)
            if t.exit_reason == "traj_time_limit":
                removal_count += 1
            else:
                new_lines.append(line)

        if removal_count == 0:
            print(
                "No trajectories with exit_reason 'traj_time_limit' found. Nothing to remove."
            )
        else:
            confirm = input(
                f"Found {removal_count} trajectories with exit_reason 'traj_time_limit'. Remove them? (y/n): "
            )
            if confirm.strip().lower() in ("y", "yes"):
                with open(filename, "w") as f:
                    f.writelines(new_lines)
                print(f"Removed {removal_count} entries from {filename}.")
            else:
                print("Aborted removal. No changes made.")
        # After processing removal, exit without doing further analysis.
        return

    # --- Continue with analysis if not removing entries ---
    # Load the file and create a list of (trajectory, idx) tuples.
    with open(filename, "r") as f:
        data = f.readlines()

    trajectories_with_idx: list[tuple[Trajectory, int]] = []
    for idx, line in enumerate(data):
        try:
            trajectories_with_idx.append(
                (Trajectory.load_from_model_dump_json(line), idx)
            )
        except:
            print(f"Error decoding JSON for line {idx}")
            continue

    # trajectories_with_idx: list[tuple[Trajectory, int]] = [
    #     (Trajectory.load_from_model_dump_json(line), idx)
    #     for idx, line in enumerate(data)
    # ]
    trajectories = [t for t, _ in trajectories_with_idx]
    # trajectories = sorted(trajectories, key=lambda x: x.docker_image)
    num_trajectories = len(trajectories)
    print(f"Loaded {num_trajectories=} trajectories")

    # Print overall success rate
    num_success = sum(t.reward == 1 for t in trajectories)
    success_rate = num_success / num_trajectories
    print(f"Success rate: {success_rate*100:.2f} ({num_success}/{num_trajectories})")

    if num_trajectories <= 20:
        print([t.reward == 1 for t in trajectories])

    # Repo-wise success rates: compute num_solved and num_total per repo,
    # and then print the mean success rate (num_solved / num_total)
    repo_df = pd.DataFrame(
        {
            "repo": [t.ds.get("repo", t.ds.get("repo_name")) for t in trajectories],
            "success": [t.reward == 1 for t in trajectories],
        }
    )
    repo_grouped = repo_df.groupby("repo").agg(
        num_solved=("success", "sum"), num_total=("success", "count")
    )
    repo_grouped["mean_success_rate"] = (
        repo_grouped["num_solved"] / repo_grouped["num_total"]
    )
    print("Success rates by repo:")
    print(repo_grouped)

    # Count empty patches
    num_empty_patches = sum(t.output_patch == "" for t in trajectories)
    print(f"Number of empty patches: {num_empty_patches}")

    # Exit reason stats
    exit_reasons = [t.exit_reason for t in trajectories]
    print("Exit reasons:")
    print(pd.Series(exit_reasons).value_counts())

    # add pd percentile results for reward_calc_time
    # get all reward_calc_time for all entries
    reward_calc_times = [t.reward_calc_time for t in trajectories]
    print(f"\nReward calc time:")
    print(pd.Series(reward_calc_times).describe(percentiles=[0.05, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95]))

    # For trajectories with exit_reason as "traj_time_limit",
    # compute the total_time_traj from the last trajectory step,
    # and the sum of llm_exec_time and env_exec_time across all steps.
    traj_time_limit_data = []
    for t, idx in trajectories_with_idx:
        if t.exit_reason == "traj_time_limit":
            # Ensure there is at least one step.
            if not t.trajectory_steps:
                continue
            last_step = t.trajectory_steps[-1]
            total_time_traj = getattr(last_step, "total_time_traj", None)
            sum_llm_exec_time = sum(
                getattr(step, "llm_exec_time", 0) for step in t.trajectory_steps
            )
            sum_env_exec_time = sum(
                getattr(step, "env_exec_time", 0) for step in t.trajectory_steps
            )
            traj_time_limit_data.append(
                {
                    "idx": idx,
                    # "docker_image": t.docker_image,
                    "total_time_traj": total_time_traj,
                    "sum_llm_exec_time": sum_llm_exec_time,
                    "sum_env_exec_time": sum_env_exec_time,
                    "num_steps": t.num_steps,
                    "max_llm_exec_time_per_step": max(t.llm_time_by_step),
                }
            )

    if traj_time_limit_data:
        df_traj_time_limit = pd.DataFrame(traj_time_limit_data)
        print("\nTime stats for trajectories with exit_reason == 'traj_time_limit':")
        print(df_traj_time_limit)
    else:
        print("\nNo trajectories with exit_reason == 'traj_time_limit' found.")

    ## number of steps with observations ending in "Error executing command:"
    error_steps = [
        len(
            [
                step
                for step in trajectory.trajectory_steps
                if step.observation.strip().endswith("Error executing command:")
            ]
        )
        for trajectory in trajectories
    ]
    print(f"Number of steps with 'Error executing command:': {sum(error_steps)}")
    print(
        pd.Series(error_steps).describe(
            percentiles=[0.05, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95]
        )
    )
    print(f"Error executing command docker images:")
    docker_names = [
        trajectory.docker_image.split(".")[-1]
        for trajectory in trajectories
        if any(
            [
                step.observation.strip().endswith("Error executing command:")
                for step in trajectory.trajectory_steps
            ]
        )
    ]
    print(docker_names)

    # error_steps = [
    #     len(
    #         [
    #             step
    #             for step in trajectory.trajectory_steps
    #             if step.observation.strip().endswith("Error executing command:")
    #         ]
    #     )
    #     for trajectory in trajectories
    #     if trajectory.reward == 1
    # ]
    # print(f"Number of steps with 'Error executing command:': {sum(error_steps)}")
    # print(
    #     pd.Series(error_steps).describe(
    #         percentiles=[0.05, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95]
    #     )
    # )

    rewards = [traj.reward for traj in trajectories]

    steps = [traj.num_steps for traj in trajectories]

    ## print accuracy by number of steps cumsum

    new_step_rewards = sorted(
        [(s, r) for s, r in zip(steps, rewards)], key=lambda x: x[0]
    )
    new_steps = [x[0] for x in new_step_rewards]
    new_rewards = [x[1] for x in new_step_rewards]

    # plot
    import matplotlib.pyplot as plt

    plt.plot(new_steps, np.cumsum(new_rewards) / np.arange(1, len(new_rewards) + 1))
    ## counts of num_steps
    plt.plot(new_steps, np.arange(len(new_rewards)) / len(new_rewards))
    plt.xlabel("Number of steps")
    plt.ylabel("Accuracy")
    plt.savefig("accuracy_by_steps.png")

    steps = [t.num_steps for t in trajectories]
    print(f"Total steps: {sum(steps)}")
    print(
        pd.Series(steps).describe(percentiles=[0.05, 0.15, 0.25, 0.4, 0.5, 0.75, 0.95])
    )

    total_time_trajs = [t.total_time_traj for t in trajectories]
    print(f"Total time: {sum(total_time_trajs)}")
    print(pd.Series(total_time_trajs).describe(percentiles=[0.05, 0.5, 0.95]))

    total_llm_times = [t.total_llm_time for t in trajectories]
    print(f"Total LLM time: {sum(total_llm_times)}")
    print(pd.Series(total_llm_times).describe(percentiles=[0.05, 0.5, 0.95]))

    total_exec_times = [t.total_env_time for t in trajectories]
    print(f"Total exec time: {sum(total_exec_times)}")
    print(pd.Series(total_exec_times).describe(percentiles=[0.05, 0.5, 0.95]))

    correct_num_steps = [t.num_steps for t in trajectories if t.reward == 1]
    print(f"Correct number of steps: {sum(correct_num_steps)}")
    print(
        pd.Series(correct_num_steps).describe(
            percentiles=[0.05, 0.5, 0.65, 0.75, 0.85, 0.95]
        )
    )
    
    if not minimal:
        # llm_time_by_step = [t.llm_time_by_step for t in trajectories]
        # print(f"LLM time by step:")
        # llm_time_by_step_df = pd.DataFrame(llm_time_by_step)
        # print(llm_time_by_step_df.describe(percentiles=[0.05, 0.5, 0.95]).round(0))

        patch_sizes = [len(t.true_output_patch) for t in trajectories]
        print(f"Total patch size: {sum(patch_sizes)}")
        print(pd.Series(patch_sizes).describe(percentiles=[0.05, 0.5, 0.95]))

        correct_num_steps = [t.num_steps for t in trajectories if t.reward == 1]
        print(f"Correct number of steps: {sum(correct_num_steps)}")
        print(
            pd.Series(correct_num_steps).describe(
                percentiles=[0.05, 0.5, 0.65, 0.75, 0.85, 0.95]
            )
        )

        correct_patch_sizes = [
            len(t.true_output_patch) for t in trajectories if t.reward == 1
        ]
        print(f"Correct patch size: {sum(correct_patch_sizes)}")
        print(pd.Series(correct_patch_sizes).describe(percentiles=[0.05, 0.5, 0.95]))

        correct_gt_patch_sizes = [
            len(t.gt_patch) for t in trajectories if t.reward == 1
        ]
        print(f"Correct gt patch size: {sum(correct_gt_patch_sizes)}")
        print(pd.Series(correct_gt_patch_sizes).describe(percentiles=[0.05, 0.5, 0.95]))

        patch_len_diff = [t.patch_len_diff for t in trajectories if t.reward == 1]
        print(f"Patch len diff when correct: {sum(patch_len_diff)}")
        print(
            pd.Series(patch_len_diff).describe(
                percentiles=[0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
            )
        )

        patch_len_diff = [
            t.patch_len_diff
            for t in trajectories
            if t.reward == 0 and t.true_output_patch
        ]
        print(f"Patch len diff when incorrect: {sum(patch_len_diff)}")
        print(
            pd.Series(patch_len_diff).describe(
                percentiles=[0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
            )
        )

        correct_patch_num_lines = [
            (t.true_num_lines_edited)
            for t in trajectories
            if t.reward == 0 and t.true_output_patch
        ]
        print(f"Correct num lines edited: {sum(correct_patch_num_lines)}")
        print(
            pd.Series(correct_patch_num_lines).describe(percentiles=[0.05, 0.5, 0.95])
        )

        correct_gt_num_lines = [
            t.gt_num_lines_edited
            for t in trajectories
            if t.reward == 0 and t.true_output_patch
        ]
        print(f"Correct gt num lines edited: {sum(correct_gt_num_lines)}")
        print(pd.Series(correct_gt_num_lines).describe(percentiles=[0.05, 0.5, 0.95]))

        correct_lines_edited_diff = [
            t.num_lines_diff
            for t in trajectories
            if t.reward == 0 and t.true_output_patch
        ]
        print(f"Num lines edited diff: {sum(correct_lines_edited_diff)}")
        print(
            pd.Series(correct_lines_edited_diff).describe(
                percentiles=[0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
            )
        )

        num_files_modified = [t.num_files_modified for t in trajectories]
        print(f"Num Files modified Pred: {sum(num_files_modified)}")
        print(
            pd.Series(num_files_modified).describe(
                percentiles=[0.1, 0.2, 0.8, 0.9, 0.95, 0.98]
            )
        )

        num_files_modified_gt = [t.num_files_modified_gt for t in trajectories]
        print(f"Num Files modified GT: {sum(num_files_modified_gt)}")
        print(
            pd.Series(num_files_modified_gt).describe(
                percentiles=[0.1, 0.2, 0.8, 0.9, 0.95, 0.98]
            )
        )

        num_files_modified_solved = [
            t.num_files_modified for t in trajectories if t.reward == 1
        ]
        print(f"Num Files modified (Rew=1): {sum(num_files_modified_solved)}")
        print(
            pd.Series(num_files_modified_solved).describe(
                percentiles=[0.1, 0.2, 0.8, 0.9, 0.95, 0.98]
            )
        )

        same_files_modified = [t.same_files_modified for t in trajectories]
        print(f"Same files modified: {sum(same_files_modified)}")
        print(pd.Series(same_files_modified).describe())

        same_files_modified = [
            t.same_files_modified
            for t in trajectories
            if t.parsed_pred_commit.get_file_name_list()
        ]
        print(f"Same files modified ignoring empty patches: {sum(same_files_modified)}")
        print(pd.Series(same_files_modified).describe())

        subset_files_modified = [
            t.subset_modified
            for t in trajectories
            if t.parsed_pred_commit.get_file_name_list()
        ]
        print(
            f"Strict Subset files modified ignoring empty patches: {sum(subset_files_modified)}"
        )
        print(pd.Series(subset_files_modified).describe())

        superset_files_modified = [
            t.superset_modified
            for t in trajectories
            if t.parsed_pred_commit.get_file_name_list()
        ]
        print(
            f"Strict Superset files modified ignoring empty patches: {sum(superset_files_modified)}"
        )

        same_files_modified = [
            t.same_files_modified for t in trajectories if t.reward == 1
        ]
        print(f"Same files modified (REWARD==1): {sum(same_files_modified)}")
        print(pd.Series(same_files_modified).describe())

        subset_files_modified = [
            t.subset_modified for t in trajectories if t.reward == 1
        ]
        print(f"Subset files modified (REWARD==1): {sum(subset_files_modified)}")

        superset_files_modified = [
            t.superset_modified for t in trajectories if t.reward == 1
        ]

        print(f"Superset files modified (REWARD==1): {sum(superset_files_modified)}")

        for t in trajectories:
            if t.reward == 1:
                if not t.same_files_modified:
                    print(t.docker_image)
                    print("GT files changed ", t.parsed_gt_commit.get_file_name_list())
                    print(
                        "Pre files changed ", t.parsed_pred_commit.get_file_name_list()
                    )
                    # if (
                    #     "toml" in t.parsed_pred_commit.file_extension_set
                    #     or "cfg" in t.parsed_pred_commit.file_extension_set
                    # ):
                    #     print(t.output_patch)
                    #     input()
                    print("#" * 30)

        ## plot cumulative success rates and save as a.png
        success_rates = [
            sum([t.reward == 1 for t in trajectories[:i]]) / i
            for i in range(1, num_trajectories)
        ]
        success_rates = [0] + success_rates

        import matplotlib.pyplot as plt

        plt.plot(success_rates)
        plt.xlabel("Number of trajectories")
        plt.ylabel("Success rate")
        plt.ylim(0.1, 0.65)
        plt.title("Cumulative success rate")

        plt.grid()

        plt.savefig("success_rate.png")
        print("Saved success_rate.png")

        plt.close()

        cummulative_num_solved = [
            sum([t.reward == 1 for t in trajectories[:i]])
            for i in range(1, num_trajectories)
        ]
        cummulative_num_solved = [0] + cummulative_num_solved
        plt.plot(cummulative_num_solved)
        plt.xlabel("Number of trajectories")
        plt.ylabel("Number of solved")
        plt.title("Cummulative number of solved")
        plt.grid()
        plt.savefig("cummulative_num_solved.png")
        print("Saved cummulative_num_solved.png")

        all_actions = [
            step.action
            for trajectory in trajectories
            for step in trajectory.trajectory_steps
        ]
        all_actions = [Action.from_string(action) for action in all_actions if action]
        all_fileeditor_view_paths = [
            action.parameters.get("path")
            for action in all_actions
            if action.function_name == "file_editor"
            and action.parameters.get("command") == "view"
            and action.parameters.get("path")
            and "." in action.parameters.get("path").split("/")[-1]
        ]
        all_extensions = [path.split(".")[-1] for path in all_fileeditor_view_paths]
        print("View Extensions:")
        print(pd.Series(all_extensions).value_counts())

        all_fileeditor_edit_paths = [
            action.parameters.get("path")
            for action in all_actions
            if action.function_name == "file_editor"
            and action.parameters.get("command") == "str_replace"
            and action.parameters.get("path")
            and "." in action.parameters.get("path").split("/")[-1]
        ]
        all_extensions = [path.split(".")[-1] for path in all_fileeditor_edit_paths]
        print("Edit Extensions:")
        print(pd.Series(all_extensions).value_counts())

        reproduce_py_edits = [
            "reproduce_issue.py" in path for path in all_fileeditor_edit_paths
        ]
        print(
            f"Edit reproduce_issue.py: {sum(reproduce_py_edits)} times (i.e. `str_replace` not `create`)"
        )

        top_edited_files = (
            pd.Series(all_fileeditor_edit_paths)
            .str.split("/")
            .str[-1]
            .value_counts()[:5]
        )
        print("Top edited files:")
        print(top_edited_files)

        count = sum(
            any(
                [
                    "ERROR: Unhandled exception" in obs
                    for obs in file_editor_observations
                ]
            )
            for trajectory in trajectories
            for file_editor_observations in [
                [
                    step.observation
                    for step in trajectory.trajectory_steps
                    if Action.from_string(step.action).function_name == "file_editor"
                ]
            ]
        )
        print(f"Unhandled file_editor exception: {count}")
        docker_images = [
            trajectory.docker_image.split(".")[-1]
            for trajectory in trajectories
            for file_editor_observations in [
                [
                    step.observation
                    for step in trajectory.trajectory_steps
                    if Action.from_string(step.action).function_name == "file_editor"
                ]
            ]
            if any(
                [
                    "ERROR: Unhandled exception" in obs
                    for obs in file_editor_observations
                ]
            )
        ]
        print(docker_images)

        avg_editor_ranges = [
            np.mean(traj.editor_view_range_lengths)
            for traj in trajectories
            if traj.editor_view_range_lengths
        ]
        print(f"Avg editor ranges: {np.mean(avg_editor_ranges)}")
        print(
            pd.Series(avg_editor_ranges).describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
            )
        )

        min_editor_ranges = [
            np.min(traj.editor_view_range_lengths)
            for traj in trajectories
            if traj.editor_view_range_lengths
        ]
        print(f"Min editor ranges: {np.mean(min_editor_ranges)}")
        print(
            pd.Series(min_editor_ranges).describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
            )
        )

        max_editor_ranges = [
            np.max(traj.editor_view_range_lengths)
            for traj in trajectories
            if traj.editor_view_range_lengths
        ]
        print(f"Max editor ranges: {np.mean(max_editor_ranges)}")
        print(
            pd.Series(max_editor_ranges).describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
            )
        )

        # correlation with reward
        editor_ranges = [
            (
                np.min(traj.editor_view_range_lengths)
                if traj.editor_view_range_lengths
                else 0
            )
            for traj in trajectories
        ]

    max_file_view_counts = [traj.max_file_view_count for traj in trajectories]
    print(f"Max View Counts: {sum(max_file_view_counts)}")
    print(
        pd.Series(max_file_view_counts).describe(
            percentiles=[0.5, 0.95, 0.97, 0.98, 0.99]
        )
    )

    max_file_view_counts = [
        traj.max_file_view_count for traj in trajectories if traj.reward == 1
    ]
    print(f"Max View Counts when correct: {sum(max_file_view_counts)}")
    print(
        pd.Series(max_file_view_counts).describe(
            percentiles=[0.5, 0.95, 0.97, 0.98, 0.99]
        )
    )

    has_bad_editor_path = [
        trajectory.has_bad_editor_path for trajectory in trajectories
    ]
    print(f"Bad editor path: {sum(has_bad_editor_path)}")
    print(pd.Series(has_bad_editor_path).mean() * 100)

    has_bad_path = [trajectory.has_bad_path for trajectory in trajectories]
    print(f"Bad editor path: {sum(has_bad_path)}")
    print(pd.Series(has_bad_path).mean() * 100)

    qwentokens = [traj.qwentokendistribution for traj in trajectories]

    token_usage_per_traj = [{k: sum(v) for k, v in x.items()} for x in qwentokens]
    token_usage_df = pd.DataFrame(token_usage_per_traj)
    print("Token usage avg:")
    print(token_usage_df.describe(percentiles=[0.05, 0.1, 0.5, 0.8, 0.9, 0.95]))

    token_usage_per_traj = [{k: max(v) for k, v in x.items()} for x in qwentokens]
    token_usage_df = pd.DataFrame(token_usage_per_traj)
    print("Token usage max:")
    print(token_usage_df.describe(percentiles=[0.05, 0.1, 0.5, 0.8, 0.9, 0.95]))

    bash_lines_to_tokens = sum(
        [traj.bash_lines_to_qwentokens for traj in trajectories], []
    )
    ## remove elements with tokens < 3000 and print #line stats
    bash_lines_to_tokens = [
        x["lines"] for x in bash_lines_to_tokens if x["tokens"] > 3000
    ]
    print("Bash lines to tokens:")
    print(
        pd.Series(bash_lines_to_tokens).describe(
            percentiles=[0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
        )
    )
    observation_tokens_by_type = [
        {
            "action": Action.from_string(step.action).function_name,
            "count": count,
        }
        for traj, tokenstats in zip(trajectories, qwentokens)
        for step, count in zip(traj.trajectory_steps, tokenstats["observation"])
    ]
    observation_tokens_by_type_df = pd.DataFrame(observation_tokens_by_type)
    ## group by action and stats
    print("Observation tokens by type:")
    print(
        observation_tokens_by_type_df.groupby("action")["count"].describe(
            percentiles=[0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1]
        )
    )

    file_editor_over_4ktokens = [
        {
            "action": Action.from_string(step.action).parameters.get("command"),
            "count": count,
        }
        for traj, tokenstats in zip(trajectories, qwentokens)
        for step, count in zip(traj.trajectory_steps, tokenstats["observation"])
        if Action.from_string(step.action).function_name == "file_editor"
        and Action.from_string(step.action).parameters.get("command")
    ]
    file_editor_over_4ktokens_df = pd.DataFrame(file_editor_over_4ktokens)
    print("File editor tokens over 4k:")
    print(
        file_editor_over_4ktokens_df.groupby("action")["count"].describe(
            percentiles=[0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1]
        )
    )

    viewer_lines_chars_tokens = [
        {
            "lines": len(step.observation.split("\n")),
            "chars": len(step.observation),
            "tokens": obstokens,
        }
        for traj, tokenstats in zip(trajectories, qwentokens)
        for step, obstokens in zip(traj.trajectory_steps, tokenstats["observation"])
        if Action.from_string(step.action).function_name == "file_editor"
    ]
    viewer_lines_chars_tokens_df = pd.DataFrame(viewer_lines_chars_tokens)
    print("Viewer lines chars tokens:")
    print(
        viewer_lines_chars_tokens_df.describe(
            percentiles=[0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
        )
    )
    ## tokens > 4k other stats
    viewer_lines_chars_tokens_df = viewer_lines_chars_tokens_df[
        viewer_lines_chars_tokens_df["tokens"] > 2500
    ]
    print("Viewer lines chars tokens > 4k:")
    print(
        viewer_lines_chars_tokens_df.describe(
            percentiles=[0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
        )
    )
    badoutputparams = [
        (
            Action.from_string(step.action).parameters,
            traj.docker_image,
            step.observation,
        )
        for traj in trajectories
        for step in traj.trajectory_steps
        if Action.from_string(step.action).function_name == "file_editor"
        if len(step.observation) > 15000
    ]
    print("Bad output params:")
    for x in badoutputparams[:10]:
        print(x[0])
        print(x[1])

    return trajectories


if __name__ == "__main__":
    fire.Fire(analyze_log, serialize=lambda x: None)
