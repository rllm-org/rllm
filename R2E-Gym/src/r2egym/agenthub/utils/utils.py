import json
import glob
from r2egym.agenthub.utils.log import get_logger
import openai
import re
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from fire import Fire
from r2egym.commit_models.diff_classes import ParsedCommit
import numpy as np
from huggingface_hub import create_repo, upload_folder, HfFolder
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
from r2egym.agenthub import SUPPORTED_REPOS
import concurrent.futures

##############################################################################
# Initialize Logger
##############################################################################
logger = get_logger(__name__)  # Initialize the logger


##############################################################################
# util fn
##############################################################################
def read_json(file_path):
    """
    Reads a JSON file and returns the parsed data.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict or list: Parsed data from the JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def read_jsonl(gpath: str) -> List[Dict]:
    """Reads a JSONL file and returns the data as a list of dictionaries."""
    data = []
    for path in glob.glob(gpath):
        with open(path, "r") as file:
            for line in file:
                try:
                    data.append(json.loads(line))
                except:
                    continue
    return data


def read_jsonl_lines(gpath: str) -> List[Dict]:
    """Reads a JSONL file and returns the data as a list of dictionaries."""
    data = []
    for path in glob.glob(gpath):
        with open(path, "r") as file:
            for line in file:
                data.append((line))
    return data


def print_results_mt(path: str, max_workers: int = 4) -> None:
    """
    Processes JSONL entries from the given path using multiple threads to compute recall metrics.

    Args:
        path (str): The file path to the JSONL data.
        max_workers (int, optional): The maximum number of threads to use. Defaults to None, which uses
                                     the number of processors on the machine multiplied by 5.
    """
    data = read_jsonl(path)
    logger.info(f"Total entries: {len(data)}")
    logger.info("-" * 50)

    # Initialize lists for storing recall metrics
    recall_all_list = []
    recall_nontest_list = []
    recall_test_list = []

    recall_all_single_nontest_list = []
    recall_nontest_single_nontest_list = []
    recall_test_single_nontest_list = []

    # Define a worker function to process each entry
    def process_entry(entry):
        try:
            docker_image = entry.get("docker_image", "")
            output_action = entry.get("output_action", "")
            trajectory = entry.get("trajectory", [])
            pred_files = entry.get("pred_files")

            # Get parsed commit information
            parsed_commit = get_parsed_commit(docker_image)

            # Extract ground truth file paths
            gt_files = [fd.path for fd in parsed_commit.file_diffs]
            gt_nontest_files = [
                fd.path for fd in parsed_commit.file_diffs if not fd.is_test_file
            ]
            gt_test_files = [
                fd.path for fd in parsed_commit.file_diffs if fd.is_test_file
            ]

            # If pred_files not given in entry, derive from output_action
            if pred_files is None:
                # Assuming first line of output_action is not a file. Adjust if needed.
                pred_files = output_action.split("\n")[1:] if output_action else []

            # Normalize paths
            gt_files = normalize_paths(gt_files)
            gt_nontest_files = normalize_paths(gt_nontest_files)
            gt_test_files = normalize_paths(gt_test_files)
            pred_files = normalize_paths(pred_files)

            # Compute recalls
            recall_all = (
                np.mean([f in pred_files for f in gt_files]) if gt_files else 1.0
            )
            recall_nontest = (
                np.mean([f in pred_files for f in gt_nontest_files])
                if gt_nontest_files
                else 1.0
            )
            recall_test = (
                np.mean([f in pred_files for f in gt_test_files])
                if gt_test_files
                else 1.0
            )

            # Thread-safe appending to shared lists
            recall_all_list.append(recall_all)
            recall_nontest_list.append(recall_nontest)
            recall_test_list.append(recall_test)

            if len(gt_nontest_files) == 1:
                recall_all_single_nontest_list.append(recall_all)
                recall_nontest_single_nontest_list.append(recall_nontest)
                recall_test_single_nontest_list.append(recall_test)

            # Logging details
            logger.info(f"Docker Image: {docker_image}")
            logger.info(f"Total Steps in Trajectory: {len(trajectory)}")
            logger.info(f"Final Output Action:\n{output_action}")
            logger.info("GT Files:\n" + "\n".join(gt_files))
            logger.info("GT Non-test Files:\n" + "\n".join(gt_nontest_files))
            logger.info("GT Test Files:\n" + "\n".join(gt_test_files))
            logger.info("Predicted Files:\n" + "\n".join(pred_files))
            logger.info(f"Recall (All): {recall_all:.4f}")
            logger.info(f"Recall (Non-test): {recall_nontest:.4f}")
            logger.info(f"Recall (Test): {recall_test:.4f}")
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Error processing entry: {e}")

    # Use ThreadPoolExecutor to process entries concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        executor.map(process_entry, data)

    # After all entries have been processed, compute and log the summary
    if recall_all_list:
        logger.info(f"Total Entries Processed: {len(recall_all_list)}")
        logger.info("===== Summary of Recalls (All Entries) =====")
        logger.info(f"Average Recall (All): {np.mean(recall_all_list):.4f}")
        logger.info(f"Average Recall (Non-test): {np.mean(recall_nontest_list):.4f}")
        logger.info(f"Average Recall (Test): {np.mean(recall_test_list):.4f}")

        if recall_all_single_nontest_list:
            logger.info("===== Summary of Recalls (Entries with 1 Non-test File) =====")
            logger.info(
                f"Average Recall (All): {np.mean(recall_all_single_nontest_list):.4f}"
            )
            logger.info(
                f"Average Recall (Non-test): {np.mean(recall_nontest_single_nontest_list):.4f}"
            )
            logger.info(
                f"Average Recall (Test): {np.mean(recall_test_single_nontest_list):.4f}"
            )
    else:
        logger.info("No entries processed for recall computation.")


def print_results(path: str) -> None:
    data = read_jsonl(path)
    logger.info(f"Total entries: {len(data)}")
    logger.info("-" * 50)

    # Lists for storing recalls of all entries
    recall_all_list = []
    recall_nontest_list = []
    recall_test_list = []

    # Lists for storing recalls of entries where number of non-test files == 1
    recall_all_single_nontest_list = []
    recall_nontest_single_nontest_list = []
    recall_test_single_nontest_list = []

    for entry in data[:]:
        docker_image = entry.get("docker_image", "")
        output_action = entry.get("output_action", "")
        trajectory = entry.get("trajectory", [])
        pred_files = entry.get("pred_files")

        # Get parsed commit information
        parsed_commit = get_parsed_commit(docker_image)

        # Extract ground truth file paths
        gt_files = [fd.path for fd in parsed_commit.file_diffs]
        gt_nontest_files = [
            fd.path for fd in parsed_commit.file_diffs if not fd.is_test_file
        ]
        gt_test_files = [fd.path for fd in parsed_commit.file_diffs if fd.is_test_file]

        # If pred_files not given in entry, derive from output_action
        if pred_files is None:
            # Assuming first line of output_action is not a file. Adjust if needed.
            pred_files = output_action.split("\n")[1:] if output_action else []

        # Normalize paths
        gt_files = normalize_paths(gt_files)
        gt_nontest_files = normalize_paths(gt_nontest_files)
        gt_test_files = normalize_paths(gt_test_files)
        pred_files = normalize_paths(pred_files)

        # Compute recalls
        recall_all = np.mean([f in pred_files for f in gt_files]) if gt_files else 1.0
        recall_nontest = (
            np.mean([f in pred_files for f in gt_nontest_files])
            if gt_nontest_files
            else 1.0
        )
        recall_test = (
            np.mean([f in pred_files for f in gt_test_files]) if gt_test_files else 1.0
        )

        # Append to overall lists
        recall_all_list.append(recall_all)
        recall_nontest_list.append(recall_nontest)
        recall_test_list.append(recall_test)

        # If there is exactly one non-test file, store these recalls in separate lists
        if len(gt_nontest_files) == 1:
            recall_all_single_nontest_list.append(recall_all)
            recall_nontest_single_nontest_list.append(recall_nontest)
            recall_test_single_nontest_list.append(recall_test)

        # Logging details
        logger.info(f"Docker Image: {docker_image}")
        logger.info(f"Total Steps in Trajectory: {len(trajectory)}")
        logger.info(f"Final Output Action:\n{output_action}")
        logger.info("GT Files:\n" + "\n".join(gt_files))
        logger.info("GT Non-test Files:\n" + "\n".join(gt_nontest_files))
        logger.info("GT Test Files:\n" + "\n".join(gt_test_files))
        logger.info("Predicted Files:\n" + "\n".join(pred_files))
        logger.info(f"Recall (All): {recall_all:.4f}")
        logger.info(f"Recall (Non-test): {recall_nontest:.4f}")
        logger.info(f"Recall (Test): {recall_test:.4f}")
        logger.info("-" * 50)

    # After processing all entries, print the average recalls
    if recall_all_list:
        # also print the total number of entries processed
        logger.info(f"Total Entries Processed: {len(recall_all_list)}")
        logger.info("===== Summary of Recalls (All Entries) =====")
        logger.info(f"Average Recall (All): {np.mean(recall_all_list):.4f}")
        logger.info(f"Average Recall (Non-test): {np.mean(recall_nontest_list):.4f}")
        logger.info(f"Average Recall (Test): {np.mean(recall_test_list):.4f}")

        # If we have entries with exactly one non-test file, summarize them
        if recall_all_single_nontest_list:
            logger.info("===== Summary of Recalls (Entries with 1 Non-test File) =====")
            logger.info(
                f"Average Recall (All): {np.mean(recall_all_single_nontest_list):.4f}"
            )
            logger.info(
                f"Average Recall (Non-test): {np.mean(recall_nontest_single_nontest_list):.4f}"
            )
            logger.info(
                f"Average Recall (Test): {np.mean(recall_test_single_nontest_list):.4f}"
            )
    else:
        logger.info("No entries processed for recall computation.")


# def get_gt_files(commit):
#     """
#     Gets the ground truth modified files for the given commit
#     """
#     gt_path = f'./commit_data/sympy/{commit}.json'
#     gt = read_json(gt_path)
#     modified_files = [x['header']['file']['path'] for x in gt['file_diffs']]
#     return modified_files


def normalize_paths(file_list):
    """Normalize file paths by removing leading slashes or other differences."""
    return [file.lstrip("./") for file in file_list]


def match_dockerimage_to_repo(docker_image: str):
    repo_match = False
    for repo_name in SUPPORTED_REPOS:
        if repo_name in docker_image:
            repo = repo_name
            repo_match = True
            break
    assert repo_match, f"Repo for {docker_image} must be one of {SUPPORTED_REPOS}"
    return repo, repo_match


def get_parsed_commit(docker_image: str) -> Optional[ParsedCommit]:
    """
    Retrieves the ParsedCommit object either from a local JSON file or directly from the Docker image.

    Args:
        docker_image (str): The Docker image name with tag.

    Returns:
        Optional[ParsedCommit]: The ParsedCommit instance if successful, else None.
    """
    # match the repo
    repo, _ = match_dockerimage_to_repo(docker_image)
    commit = docker_image.split(":")[-1]
    commit_path = f"./commit_data/{repo}/{commit}.json"

    # Check if the JSON file exists locally
    if os.path.isfile(commit_path):
        try:
            with open(commit_path, "r", encoding="utf-8") as f:
                parsed_commit = ParsedCommit(**json.load(f))
            print(f"Loaded parsed_commit from local file: {commit_path}")
            return parsed_commit
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing local JSON file {commit_path}: {e}")
            # Optionally, proceed to fetch from Docker if local file is corrupted
        except Exception as e:
            print(f"Unexpected error reading local file {commit_path}: {e}")
            return None

    # If the file doesn't exist locally, extract it from the Docker image
    try:
        # Specify the path to parsed_commit.json inside the Docker container
        # Adjust '/app/parsed_commit.json' based on your container's file structure
        docker_command = [
            "docker",
            "run",
            "--rm",
            docker_image,
            "cat",
            f"/{repo}/parsed_commit.json",
        ]

        print(f"Running Docker command to extract JSON from image: {docker_image}")
        json_content = subprocess.check_output(docker_command, stderr=subprocess.STDOUT)
        json_str = json_content.decode("utf-8")

        # Parse the JSON content into ParsedCommit
        parsed_commit = ParsedCommit(**json.loads(json_str))
        print(f"Successfully extracted parsed_commit from Docker image: {docker_image}")
        return parsed_commit

    except subprocess.CalledProcessError as e:
        print(
            f"Error executing Docker command for {docker_image}: {e.output.decode('utf-8')}"
        )
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from Docker image {docker_image}: {e}")
    except TypeError as e:
        print(f"Error initializing ParsedCommit from JSON data for {docker_image}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {docker_image}: {e}")

    return None


# def get_parsed_commit_old(docker_image):
#     commit = docker_image.split(':')[-1]
#     commit_path = f'./commit_data/sympy/{commit}.json'
#     with open(commit_path) as f:
#         parsed_commit = ParsedCommit(**json.load(f))
#     return parsed_commit


def push_model_to_hf_hub(
    local_model_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Add model and tokenizer",
):
    """
    Load a model & tokenizer from a local path, then push both to the Hugging Face Hub.

    Parameters:
    -----------
    local_model_path : str
        Path to the folder containing your final model files
        (e.g., config.json, tokenizer.json, model weights, etc.).
    repo_id : str
        The name of the repository on the Hugging Face Hub (e.g., "my-username/my-new-model").
    private : bool, optional
        Whether the repository should be private (default=False).
    commit_message : str, optional
        Commit message for both the model and tokenizer push (default="Add model and tokenizer").
    """
    logger.info(f"loading model from {local_model_path}")
    # 1. Load the model & tokenizer from the local directory
    model = AutoModelForCausalLM.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    # 2. Push the model to the specified repo
    logger.info(f"pushing model to {repo_id}")
    full_repo_id = f"r2e-edits/{repo_id}"
    model.push_to_hub(
        repo_id=full_repo_id, private=private, commit_message=commit_message
    )

    # 3. Push the tokenizer to the same repo
    tokenizer.push_to_hub(
        repo_id=full_repo_id, private=private, commit_message=commit_message
    )
    print(f"Model and tokenizer pushed to: https://huggingface.co/{repo_id}")


def push_model_to_hf_hub_old(
    local_model_path, repo_id, private=False, commit_message="Add model"
):
    """
    Push a local model folder to the Hugging Face Hub.

    Parameters:
    -----------
    local_model_path : str
        Path to the folder containing model files (e.g. checkpoint, config, tokenizer, etc.).
    repo_id : str
        The name of the repository on the Hugging Face Hub (format: 'user_or_org/repo_name').
    private : bool, optional
        Whether the repository should be private, default is True.
    commit_message : str, optional
        Commit message used when uploading, default is "Add model".
    """
    full_repo_id = f"r2e-edits/{repo_id}"

    # Create (or reuse) the repo on the Hub
    create_repo(
        repo_id=full_repo_id,
        token=os.environ.get("HF_TOKEN"),
        private=private,
        exist_ok=True,
    )

    # Upload your local model folder to the repo
    upload_folder(
        folder_path=local_model_path,
        path_in_repo="",  # If you want to store under a subdirectory, specify it here
        repo_id=full_repo_id,
        token=os.environ.get("HF_TOKEN"),
        commit_message=commit_message,
    )
    print(f"Model pushed to: https://huggingface.co/{repo_id}")


##############################################################################
# main fn
##############################################################################
if __name__ == "__main__":
    # Expose all functions to the CLI using Fire
    Fire()
