import json
from datetime import datetime

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.commit_models.parse_diff import CommitParser
from r2egym.agenthub.action import Action
from r2egym.agenthub.trajectory.swebench_utils import (
    swebench_report,
    swebench_parse,
    PASS_TO_PASS,
)


# ##############################################################################
# # TrajectoryStep Dataclass for per step stats
# ##############################################################################
class TrajectoryStep(BaseModel):
    step_idx: int

    ## key parts
    thought: str
    action: str
    observation: str
    done: bool
    info: dict

    ## tokens
    token_usage_prompt: int
    token_usage_completion: int
    token_usage_total: int

    ## metadata (current step stats)
    llm_exec_time: float
    env_exec_time: float
    total_step_time: float
    total_time_traj: float
    step_count: int

    @property
    def parsed_action(self):
        return Action.from_string(self.action)


# ##############################################################################
# # Trajectory Dataclass for overall traj stats
# ##############################################################################
class Trajectory(BaseModel):
    # Tell Pydantic it can accept actual model instances:
    # model_config = ConfigDict(arbitrary_types_allowed=True)

    ##############################
    # Trajectory steps data
    ##############################
    trajectory_steps: list[TrajectoryStep]

    ##############################
    # problem metadata
    ##############################
    problem_statement: str
    docker_image: str
    exp_name: Optional[str] = None  # experiment name

    ##############################
    # Agent and Environment Args
    ##############################
    env_args: Dict[str, Any] = Field(default_factory=dict)
    agent_args: Dict[str, Any] = Field(default_factory=dict)
    ds: Optional[dict] = None  # data entry from swebench or r2e hf datasets

    ##############################
    # Limits
    ##############################
    # steps
    max_steps: int
    max_steps_absolute: int
    # tokens
    max_token_limit: int
    # time
    max_llm_time: int  # per query
    max_exec_time: int  # per env execution
    max_total_time: int  # overall agent run limit

    ##############################
    # Outcome
    ##############################
    exit_reason: str  # reason for exit
    output_patch: str  # final output patch
    # outputs after test execution [Optional]
    reward: Optional[float] = None  # success for editing agent
    reward_calc_time: Optional[float] = None  # time taken to calculate reward
    test_output: Optional[str] = None  # output after test execution

    ##############################
    # Evaluation
    ##############################
    custom_test_outputs: dict = {}  # custom test outputs
    regression_test_output: Optional[str] = None  # regression test output
    verifier_prob: Optional[float] = None  # verifier yes probability
    reproduction_test_scores: list[int] = []  # reproduction test score

    @classmethod
    def load_from_model_dump_json(cls, json_string: str):
        return Trajectory.model_validate_json(json_string)

    @property
    def instance_name(self):
        return self.docker_image.split(".")[-1]

    @property
    def total_time_traj(self):
        return self.trajectory_steps[-1].total_time_traj

    @property
    def num_steps(self):
        return len(self.trajectory_steps)

    @property
    def num_tokens_prompt(self):
        return sum([step.token_usage_prompt for step in self.trajectory_steps])

    @property
    def num_tokens_completion(self):
        return sum([step.token_usage_completion for step in self.trajectory_steps])

    @property
    def num_tokens_total(self):
        return sum([step.token_usage_total for step in self.trajectory_steps])

    @property
    def total_llm_time(self):
        return sum([step.llm_exec_time for step in self.trajectory_steps])

    @property
    def total_env_time(self):
        return sum([step.env_exec_time for step in self.trajectory_steps])

    @property
    def llm_time_by_step(self):
        return [step.llm_exec_time for step in self.trajectory_steps]

    @property
    def pass_1(self):
        return self.reward == 1

    @property
    def swebench_results_dict(self):
        return swebench_report(self.ds, self.test_output)

    @property
    def swebench_log_parse(self):
        return swebench_parse(self.ds, self.test_output)

    @property
    def p2p_rate(self):
        return self.swebench_results_dict[PASS_TO_PASS]["success"]

    @property
    def p2p_count(self):
        return len(self.p2p_rate)

    @property
    def regression_pass_count(self):
        score = swebench_parse(self.ds, self.regression_test_output)
        return len([x for x, y in score.items() if y == "PASSED"])

    @property
    def regression_parse(self):
        return swebench_parse(self.ds, self.regression_test_output)

    @property
    def default_test_count(self):
        score = swebench_parse(self.ds, self.test_output)
        return len(score)

    @property
    def regression_test_count(self):
        score = swebench_parse(self.ds, self.regression_test_output)
        return len(score)

    @property
    def get_df_dict(self):
        try:
            patch_size = len(self.true_output_patch)
        except Exception as e:
            patch_size = len(self.output_patch)

        try:
            patch_num_lines = self.true_num_lines_edited
        except Exception as e:
            patch_num_lines = 100

        try:
            p2p_rate = self.p2p_rate
        except:
            p2p_rate = [0]

        return {
            "docker_image": self.docker_image,
            "pass@1": self.pass_1,
            "step_count": self.num_steps,
            "patch": self.true_output_patch,
            "patch_size": patch_size,
            "patch_num_lines": patch_num_lines,
            "created_files": self.created_files,
            "edited_files": self.editor_files,
            "viewer_files": self.viewer_files,
            "num_created_files": len(self.created_files),
            "p2p_rate": len(p2p_rate),
            "regression_pass_count": self.regression_pass_count,
        }

    @property
    def parsed_pred_commit(self) -> ParsedCommit:
        return (
            CommitParser().parse_commit(
                "a", "b", self.output_patch, "fake", datetime.now(), None
            )
            if self.ds is not None
            else None
        )

    @property
    def parsed_gt_commit(self) -> ParsedCommit:
        return ParsedCommit(
            **json.loads(
                self.ds.get("parsed_commit_content", self.ds.get("parsed_commit", None))
            )
        )

    @property
    def gt_patch(self):
        return self.parsed_gt_commit.get_patch(test_file=False)

    @property
    def gt_patch_dict(self):
        patch_dict = {}
        for file in self.parsed_gt_commit.get_file_name_list():
            patch_dict[file] = self.parsed_gt_commit.get_patch(file)
        return patch_dict

    @property
    def pred_patch_dict(self):
        patch_dict = {}
        for file in self.editor_files:
            patch_dict[file] = self.parsed_pred_commit.get_patch(file)
        return patch_dict

    @property
    def gt_patch_with_tests(self):
        return self.parsed_gt_commit.get_patch(test_file=False)

    @property
    def true_output_patch(self):
        try:
            return self.parsed_pred_commit.get_patch(
                include_files=self.editor_files
            )  ## note only python, pyx not covered
        except Exception as e:
            print(f"Error in true_output_patch: {e}")
            return self.output_patch

    @property
    def true_num_lines_edited(self):
        return self.parsed_pred_commit.get_num_lines_edited(
            include_files=self.editor_files
        )

    @property
    def gt_patch(self):
        return self.parsed_gt_commit.get_patch(test_file=False)

    @property
    def gt_num_lines_edited(self):
        return self.parsed_gt_commit.get_num_lines_edited(test_file=False)

    @property
    def patch_len_diff(self):
        return len(self.gt_patch) - len(self.true_output_patch)

    @property
    def num_lines_diff(self):
        if abs(self.gt_num_lines_edited - self.true_num_lines_edited) > 10:
            print(self.docker_image)
        return self.gt_num_lines_edited - self.true_num_lines_edited

    @property
    def gt_relevant_files(self):
        return (
            self.ds["relevant_files"]
            if "relevant_files" in self.ds
            else self.parsed_gt_commit.get_file_name_list()
        )

    @property
    def trajectory_modified_files(self):
        return list(
            set(self.parsed_pred_commit.get_file_name_list()).intersection(
                self.editor_files
            )
        )

    @property
    def same_files_modified(self):
        return set(self.gt_relevant_files) == set(self.trajectory_modified_files)

    @property
    def subset_modified(self):
        return (
            set(self.trajectory_modified_files).issubset(set(self.gt_relevant_files))
            and not self.same_files_modified
        )

    @property
    def superset_modified(self):
        return (
            set(self.trajectory_modified_files).issuperset(set(self.gt_relevant_files))
            and not self.same_files_modified
        )

    @property
    def num_files_modified(self):
        return len(self.trajectory_modified_files)

    @property
    def num_files_modified_gt(self):
        return len(self.gt_relevant_files)

    @property
    def viewer_files(self):
        return [
            action.parameters.get("path")
            for t in self.trajectory_steps
            for action in [Action.from_string(t.action)]
            if action.function_name == "file_viewer"
            and "." in action.parameters.get("path").split("/")[-1]
        ]

    @property
    def viewer_extensions(self):
        return list(set([path.split(".")[-1] for path in self.viewer_files]))

    @property
    def editor_files(self):
        return [
            action.parameters.get("path").replace("/testbed/", "")
            for t in self.trajectory_steps
            for action in [Action.from_string(t.action)]
            if action.function_name == "file_editor"
            and action.parameters.get("command") == "str_replace"
            and action.parameters.get("path")
            and "." in action.parameters.get("path").split("/")[-1]
        ]

    @property
    def created_files(self):
        return [
            action.parameters.get("path").replace("/testbed/", "")
            for t in self.trajectory_steps
            for action in [Action.from_string(t.action)]
            if action.function_name == "file_editor"
            and action.parameters.get("command") == "create"
            and (action.parameters.get("path"))
            and "." in action.parameters.get("path").split("/")[-1]
        ]

    @property
    def editor_extensions(self):
        return list(set([path.split(".")[-1] for path in self.editor_files]))

    @property
    def detect_test_command(self):
        assert len(self.created_files) == 1
        for t in self.trajectory_steps:
            action = Action.from_string(t.action)

            if action.function_name == "execute_bash":
                cmd = action.parameters.get("cmd")
                if cmd:
                    if "python" in cmd and self.created_files[0] in cmd:
                        return cmd
                else:
                    print("parameters ", action.parameters)
        return None

    @property
    def editor_view_range_lengths(self):
        return [
            eval(action.parameters.get("view_range"))[1]
            - eval(action.parameters.get("view_range"))[0]
            for t in self.trajectory_steps
            for action in [Action.from_string(t.action)]
            if action.function_name == "file_editor"
            and action.parameters.get("command") == "view"
            and action.parameters.get("view_range")
            and (
                (
                    action.parameters.get("concise")
                    and action.parameters.get("concise") == "False"
                )
                or not action.parameters.get("concise")
            )
        ]

    @property
    def file_viewer_view_paths(self):
        return [
            action.parameters.get("path")
            for t in self.trajectory_steps
            for action in [Action.from_string(t.action)]
            if action.function_name == "file_editor"
            and action.parameters.get("command") == "view"
            and action.parameters.get("path")
            and action.parameters.get("path").endswith(".py")
        ]

    @property
    def max_file_view_count(self):
        # max frequency in viewer_files
        # return (
        #     (
        #         max(
        #             [
        #                 self.file_viewer_view_paths.count(ext)
        #                 for ext in set(self.file_viewer_view_paths)
        #             ]
        #         )
        #     )
        #     if self.file_viewer_view_paths
        #     else 0
        # )

        # max_count = 0
        # current_count = 0
        # prev_file = None

        # for file in self.file_viewer_view_paths:
        #     if file == prev_file:
        #         current_count += 1
        #     else:
        #         current_count = 1
        #     max_count = max(max_count, current_count)
        #     prev_file = file
        # return max_count

        max_count = 0
        current_count = 0
        prev_file = None

        for t in self.trajectory_steps:
            action = Action.from_string(t.action)
            if (
                action.function_name == "file_editor"
                and action.parameters.get("command") == "view"
                and action.parameters.get("path")
                and action.parameters.get("path").endswith(".py")
            ):
                file = action.parameters.get("path")
                if file == prev_file:
                    current_count += 1
                else:
                    current_count = 1
                max_count = max(max_count, current_count)
                prev_file = file
            else:
                prev_file = None
        return max_count

    @property
    def has_bad_editor_path(self):
        # "has /testbed/testbed" in path
        return any(
            [
                "/testbed/testbed" in action.parameters.get("path")
                for t in self.trajectory_steps
                for action in [Action.from_string(t.action)]
                if action.function_name == "file_editor"
                and action.parameters.get("path")
            ]
        )

    @property
    def has_bad_path(self):
        # "has /testbed/testbed" in path
        return any(["/testbed/testbed" in t.action for t in self.trajectory_steps])

    @property
    def num_lines_bash_exec(self):
        return [
            t.observation.count("\n")
            for t in self.trajectory_steps
            for action in [Action.from_string(t.action)]
            if action.function_name == "execute_bash"
        ]

    @property
    def qwentokendistribution(self):
        import litellm

        token_count_fn = lambda x: litellm.token_counter(
            model="Qwen/Qwen2.5-32B", text=x
        )
        token_counts = {
            "action": [],
            "thought": [],
            "observation": [],
        }
        for step in self.trajectory_steps:
            token_counts["action"].append(token_count_fn(step.action))
            token_counts["thought"].append(token_count_fn(step.thought))
            token_counts["observation"].append(token_count_fn(step.observation))
        return token_counts

    @property
    def bash_lines_to_qwentokens(self):
        import litellm

        token_count_fn = lambda x: litellm.token_counter(
            model="Qwen/Qwen2.5-32B", text=x
        )
        lines_tokens = []
        for step in self.trajectory_steps:
            if Action.from_string(step.action).function_name == "execute_bash":
                lines_tokens.append(
                    {
                        "lines": step.observation.count("\n"),
                        "tokens": token_count_fn(step.observation),
                    }
                )
        return lines_tokens

    @property
    def true_output_patch_only_existing_files(self):
        try:
            return self.parsed_pred_commit.get_patch(
                include_files=set(self.editor_files) - set(self.created_files)
            )  ## note only python, pyx not covered
        except Exception as e:
            print(f"Error in true_output_patch_only_modified: {e}")
            return self.output_patch

    def swebench_reasoning_trace(self):
        reasoning_trace = ""
        for step_idx, step in enumerate(self.trajectory_steps):
            reasoning_trace += "=" * 100 + "\n"
            reasoning_trace += f"Step {step_idx}:\n\n"
            reasoning_trace += f"Thought:\n\n{step.thought}\n"
            reasoning_trace += f"Action:\n\n{step.action}\n"
            reasoning_trace += f"Observation:\n\n{step.observation}\n"
        return reasoning_trace

    def create_swebench_submission(self):
        return {
            "instance_id": self.instance_name,
            "model_name_or_path": self.exp_name,
            "model_patch": self.true_output_patch,
        }

    @property
    def reproduction_test_score(self):
        return sum(self.reproduction_test_scores)
