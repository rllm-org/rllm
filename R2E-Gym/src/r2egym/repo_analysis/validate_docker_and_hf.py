import os
import json
import time
import math
import random
import traceback
import subprocess
from threading import Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed


import tqdm
import docker
from pydantic import BaseModel

from r2e.paths import REPOS_DIR
import r2egym.repo_analysis.issues as issues
from r2egym.logging import setup_logging, Logger, INFO
from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.repo_analysis.build_syn_issue import get_prompt
from docker_bash_utils.docker_list_tags import fetch_docker_tags
from r2egym.repo_analysis.execution_log_parser import parse_log_fn, decolor_dict_keys
from r2egym.repo_analysis.execution_result_analysis import (
    ExecutionResult,
    CommitExecutionType,
)

docker_semaphore = Semaphore(15)

main_logger = setup_logging(
    name="validate_logs",
    log_file="validate_logs/main.log",
    console=True,
    level=INFO,
)


class DatasetRow(BaseModel):
    repo_name: str
    docker_image: str
    commit_hash: str

    parsed_commit_content: str
    execution_result_content: str

    modified_files: list[str]
    modified_entity_summaries: list[dict]
    relevant_files: list[str]

    num_non_test_files: int
    num_non_test_func_methods: int
    num_non_test_lines: int

    prompt: str
    problem_statement: str

    expected_output_json: str = ""

    @property
    def parsed_commit(self) -> ParsedCommit:
        return ParsedCommit(**json.loads(self.parsed_commit_content))

    @property
    def execution_result(self) -> ExecutionResult:
        return ExecutionResult(**json.loads(self.execution_result_content))


def fetch_docker_tags_for_repo(repo_name: str):
    base_image = f"namanjain12/{repo_name}_final"
    tags = fetch_docker_tags(base_image)
    docker_image_list = [f"{base_image}:{tag['name']}" for tag in tags]
    return docker_image_list


def get_issues_for_repo(repo_name: str):
    match repo_name:
        case "pandas":
            return issues.pandas_issues
        case "sympy":
            return issues.sympy_issues
        case "numpy":
            return issues.numpy_issues
        case "pillow":
            return issues.pillow_issues
        case "coveragepy":
            return issues.coveragepy_issues
        case "datalad":
            return issues.datalad_issues
        case "aiohttp":
            return issues.aiohttp_issues
        case "pyramid":
            return issues.pyramid_issues
        case "scrapy":
            return issues.scrapy_issues
        case "orange3":
            return issues.orange3_issues
        case "tornado":
            return issues.tornado_issues

    raise ValueError(f"No issues found for repo {repo_name}")


def validate_docker(
    repo_name: str, docker_image: str, ds: dict, gt_patch: str, issue_logger: Logger
):
    try:
        runtime = DockerRuntime(repo_path=f"/testbed", docker_image=docker_image, ds=ds)
    except Exception as e:
        issue_logger.error(f"Failed to initialize runtime for {docker_image}: {e}")
        issue_logger.error(traceback.format_exc())
        return False

    try:

        # Check test output before applying patch
        success_before = runtime._calculate_reward()
        assert success_before == 0.0, "the tests before applying gt patch should fail"
        issue_logger.info(f"Tests failed before applying gt patch as expected")

        # Apply the ground-truth patch
        runtime.apply_patch(gt_patch)
        issue_logger.info(f"Applied ground-truth patch")

        # Check test output after applying patch
        success_after = runtime._calculate_reward()
        assert success_after == 1.0, "the tests after applying gt patch should pass"
        issue_logger.info(f"Tests passed after applying gt patch as expected")

        return True
    except Exception as e:
        issue_logger.info(f"Validation failed for {repo_name} {docker_image}: {e}")
        issue_logger.error(traceback.format_exc())
        issue_logger.error("\n\n[run tests below]\n\n")
        run_tests_output = runtime.run_tests()[0]
        issue_logger.error(run_tests_output)
        parse = parse_log_fn(repo_name)(run_tests_output)
        parse = decolor_dict_keys(parse)
        parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
        issue_logger.error("parsed -\n" + str(parse))
        expected = json.loads(ds["expected_output_json"])
        expected = decolor_dict_keys(expected)
        expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

        for key in parse:
            if key not in expected:
                issue_logger.error(f"[[missing]] - {key} - {parse[key]}")
                continue
            if parse[key] != expected[key]:
                issue_logger.error(
                    f"[[mismatch]] - {key} - {parse[key]} - {expected[key]}"
                )
            else:
                pass
                # issue_logger.info(f"match - {key} - {parse[key]} - {expected[key]}")
        issue_logger.error("expected -\n" + str(expected))
        return False
    finally:
        runtime.close()


def revalidate_docker(
    repo_name: str,
    docker_image: str,
    old_exec_result: ExecutionResult,
    ds: dict,
    issue_logger: Logger,
):
    try:
        runtime = DockerRuntime(repo_path=f"/testbed", docker_image=docker_image, ds=ds)
    except Exception as e:
        issue_logger.error(f"Failed to initialize runtime for {docker_image}: {e}")
        issue_logger.error(traceback.format_exc())
        return None

    new_exec_result = ExecutionResult(
        repo_name=repo_name,
        new_commit_hash=old_exec_result.new_commit_hash,
        test_file_codes=old_exec_result.test_file_codes,
        test_file_names=old_exec_result.test_file_names,
        setup_res_code=old_exec_result.setup_res_code,
        setup_res_stdout=old_exec_result.setup_res_stdout,
        setup_res_stderr=old_exec_result.setup_res_stderr,
    )

    try:
        ## run tests
        stdout, stderr, _ = runtime.demux_run_tests()
        new_exec_result.old_commit_res_code = 0
        new_exec_result.old_commit_res_stdout = stdout
        new_exec_result.old_commit_res_stderr = stderr
        issue_logger.info(f"Ran tests for old commit")
        issue_logger.info(f"stdout:\n{stdout}")
        issue_logger.info(f"stderr:\n{stderr}")

        ## checkout to new commit
        runtime.checkout(new_exec_result.new_commit_hash)
        issue_logger.info(f"Checked out to new commit")

        ## run tests
        stdout, stderr, _ = runtime.demux_run_tests()
        new_exec_result.new_commit_res_code = 0
        new_exec_result.new_commit_res_stdout = stdout
        new_exec_result.new_commit_res_stderr = stderr
        issue_logger.info(f"Ran tests for new commit")
        issue_logger.info(f"stdout:\n{stdout}")
        issue_logger.info(f"stderr:\n{stderr}")
    except Exception as e:
        issue_logger.error(f"Failed to revalidate {docker_image}: {e}")
        issue_logger.error(traceback.format_exc())
        return None
    finally:
        runtime.close()

    return new_exec_result


def file_relevance_filter(
    repo_name: str,
    docker_image: str,
    ds: dict,
    parsed_commit: ParsedCommit,
    issue_logger: Logger,
):
    # try:

    # Extract non-test files from the patch
    non_test_files = [fd.path for fd in parsed_commit.file_diffs if not fd.is_test_file]

    relevant_files = []
    irrelevant_files = []

    for file_path in non_test_files:
        issue_logger.info(f"Checking relevance of {file_path}")
        if not file_path.endswith(".py"):
            irrelevant_files.append(file_path)
            issue_logger.info(f"Irrelevant file: {file_path}")
            continue

        # Initialize the Runtime directly
        runtime = DockerRuntime(repo_path=f"/testbed", docker_image=docker_image, ds=ds)

        # Create a modified patch excluding the current file
        modified_patch = parsed_commit.get_patch(
            test_file=True, non_test_file=True, exclude_files=[file_path]
        )

        # Apply the modified patch
        runtime.apply_patch(modified_patch)

        # Run tests
        success_after = runtime._calculate_reward()

        if success_after == 0.0:
            # Tests failed; the excluded file is relevant
            relevant_files.append(file_path)
            issue_logger.info(f"Relevant file: {file_path}")
        else:
            # Tests passed; the excluded file is irrelevant
            irrelevant_files.append(file_path)
            issue_logger.info(f"Irrelevant file: {file_path}")

        # close the runtime
        runtime.close()

    overall_success = len(relevant_files) > 0

    return relevant_files


def validate_docker_and_get_row(arg):
    repo_name: str
    docker_image: str
    do_validate: bool

    repo_name, docker_image, do_validate = arg

    issue_logger = setup_logging(
        name=docker_image,
        log_file=f"validate_logs/{docker_image}.log",
        console=False,
        level=INFO,
    )

    try:
        commit_hash = docker_image.split(":")[1]

        parsed_commit_file = f"commit_data/{repo_name}/{commit_hash}.json"
        with open(parsed_commit_file, "r", encoding="utf-8") as f:
            parsed_commit = ParsedCommit(**json.load(f))
            gt_patch = parsed_commit.get_patch()

        execution_result_file = (
            REPOS_DIR / f"{repo_name}_{commit_hash}" / "execution_result.json"
        )
        with open(execution_result_file, "r", encoding="utf-8") as f:
            execution_result = ExecutionResult(**json.load(f))

        expected_output = json.dumps(execution_result.new_commit_log_parse, indent=4)

        syn_issue_file = REPOS_DIR / f"{repo_name}_{commit_hash}" / f"syn_issue.json"
        with open(syn_issue_file, "r", encoding="utf-8") as f:
            syn_issue_content = json.load(f)
            syn_issue = syn_issue_content["syn_issue"]
            old_prompt = syn_issue_content["prompt"]

        ds = {
            "problem_statement": syn_issue,
            "expected_output_json": expected_output,
        }

        if do_validate:
            if docker_semaphore:
                validated = validate_docker(
                    repo_name, docker_image, ds, gt_patch, issue_logger
                )
                if validated:
                    issue_logger.info(f"Validation passed for {docker_image}")
                else:
                    issue_logger.warning(
                        f"Validation failed for {docker_image}... revalidating"
                    )
                    # return None

                    new_expected_result = revalidate_docker(
                        repo_name, docker_image, execution_result, ds, issue_logger
                    )
                    if new_expected_result is None:
                        issue_logger.error(f"Revalidation failed for {docker_image}")
                        return None
                    commit_execution_type, improved_fn_list = (
                        new_expected_result.is_good_exec()
                    )
                    if commit_execution_type != CommitExecutionType.NEW_COMMIT_BETTER:
                        issue_logger.error(f"Revalidation failed for {docker_image}")
                        return None

                    issue_logger.info(f"Revalidation passed for {docker_image}")
                    issue_logger.info(f"new improved_fn_list: {improved_fn_list}")
                    issue_logger.info(
                        f"old improved_fn_list: {execution_result.is_good_exec()[1]}"
                    )

                    all_old_exec_fns = list(
                        set(execution_result.new_commit_log_parse.keys())
                    )
                    all_new_exec_fns = list(
                        set(new_expected_result.new_commit_log_parse.keys())
                    )

                    for fn in all_old_exec_fns:
                        if fn not in all_new_exec_fns:
                            issue_logger.error(
                                f"Missing function {fn} in new docker execution"
                            )
                        elif (
                            execution_result.new_commit_log_parse[fn]
                            != new_expected_result.new_commit_log_parse[fn]
                        ):
                            issue_logger.error(
                                f"Function {fn} mismatch after new docker execution"
                            )

                    for fn in all_new_exec_fns:
                        if fn not in all_old_exec_fns:
                            issue_logger.error(
                                f"Missing function {fn} in old non-docker execution"
                            )

                    with open(execution_result_file, "w", encoding="utf-8") as f:
                        f.write(new_expected_result.model_dump_json(indent=4))

                    expected_output = json.dumps(
                        new_expected_result.new_commit_log_parse, indent=4
                    )
                    ds["expected_output_json"] = expected_output

        if docker_semaphore:
            relevant_files = file_relevance_filter(
                repo_name, docker_image, ds, parsed_commit, issue_logger
            )

        if not relevant_files:
            issue_logger.info(f"No relevant files found for {docker_image}")
            return None
        issue_logger.info(
            f"Found {len(relevant_files)} relevant files for {docker_image}"
        )

        repo_issues = get_issues_for_repo(repo_name)
        issues_string = ""
        for idx, issue in enumerate(repo_issues):
            issues_string += f"Example {idx + 1}:\n\n"
            issues_string += f"[ISSUE]\n{issue}\n\n[/ISSUE]\n\n"

        issue_logger.info(f"Built the issue string for {docker_image}")

        prompt = get_prompt(parsed_commit, execution_result)

        issue_logger.info(f"Built prompt for {docker_image}")

        # issue_logger.info(f"loaded old prompt for {docker_image}")

        row = DatasetRow(
            repo_name=repo_name,
            docker_image=docker_image,
            commit_hash=commit_hash,
            parsed_commit_content=parsed_commit.model_dump_json(indent=4),
            execution_result_content=execution_result.model_dump_json(indent=4),
            modified_files=parsed_commit.file_name_list,
            modified_entity_summaries=[
                entity.json_summary_dict() for entity in parsed_commit.edited_entities()
            ],
            relevant_files=relevant_files,
            num_non_test_files=parsed_commit.num_non_test_files,
            num_non_test_func_methods=parsed_commit.num_function_entities(False)
            + parsed_commit.num_class_entities(False),
            num_non_test_lines=parsed_commit.num_non_test_edited_lines,
            prompt=prompt,
            # old_prompt=old_prompt,
            # old_synthetic_issue=syn_issue,
            problem_statement="",
            expected_output_json=expected_output,
        )

        issue_logger.info(f"Row collected for {docker_image}")

        issue_logger.info(f"Removed image {docker_image}")
        return row
    except Exception as e:
        issue_logger.error(f"Error for {docker_image}: {e}")
        issue_logger.error(traceback.format_exc())
        return None
    finally:
        try:
            pass
            ## remove image locally
            docker.client.from_env().images.remove(docker_image, force=True)
        except Exception as e:
            pass


def pull_image_with_retries(
    client: docker.DockerClient, image: str, retries: int = 3, delay: int = 5
):
    """
    Attempt to pull a Docker image with retries.

    Args:
        client (docker.DockerClient): Docker client instance.
        image (str): Docker image name with tag.
        retries (int): Number of retry attempts.
        delay (int): Delay between retries in seconds.

    Returns:
        tuple: (image, success_flag, error_message)
    """
    issue_logger = setup_logging(
        name=image,
        log_file=f"validate_logs/{image}.log",
        console=False,
        level=INFO,
    )
    issue_logger.info(f"Attempting to pull image: {image}")

    for attempt in range(1, retries + 1):
        try:
            repo, tag = image.split(":") if ":" in image else (image, "latest")
            issue_logger.info(f"Attempting to pull image: {image} (Attempt {attempt})")
            client.images.pull(repository=repo, tag=tag)
            issue_logger.info(f"Successfully pulled image: {image}")
            return (image, True, None)
        except docker.errors.APIError as e:
            if "toomanyrequests" in str(e):
                sleep_time = (3**attempt) + random.uniform(1, 5)
                time.sleep(sleep_time)
                issue_logger.info(
                    f"Rate limited while pulling image {image}. Retrying in {sleep_time} seconds."
                )
                continue
        except Exception as e:
            issue_logger.error(f"Unexpected error for image {image}: {e}")
            return (image, False, str(e))
    issue_logger.error(f"Max retries exceeded for image {image}")
    return (image, False, "Max retries exceeded")


def pre_pull_docker_images(
    image_list, max_workers=40, retries=5, delay=5, pull_timeout=300
):
    """
    Pre-pull Docker images using multithreading with retries and progress monitoring.

    Args:
        image_list (List[str]): List of Docker images to pull.
        max_workers (int, optional): Maximum number of concurrent threads. Defaults to 10.
        retries (int, optional): Number of retry attempts for each image. Defaults to 3.
        delay (int, optional): Delay between retries in seconds. Defaults to 5.
    """
    client = docker.from_env()
    results = []
    failed_images = []

    main_logger.info(
        f"[pre_pull_docker_images] Starting to pull {len(image_list)} Docker images "
        f"with up to {max_workers} concurrent threads."
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(pull_image_with_retries, client, img, retries, delay): img
            for img in image_list
        }

        with tqdm.tqdm(
            total=len(future_to_image), desc="Pulling Docker Images"
        ) as pbar:
            for future in as_completed(future_to_image):
                image = future_to_image[future]
                try:
                    # Wait for each future with a pull_timeout
                    result = future.result(timeout=pull_timeout)
                    results.append(result)
                    if not result[1]:
                        failed_images.append((result[0], result[2]))
                except TimeoutError:
                    main_logger.info(
                        f"[pre_pull_docker_images] Timeout pulling {image}"
                    )
                    results.append((image, False, "Timeout"))
                    failed_images.append((image, "Timeout"))
                except Exception as exc:
                    main_logger.info(
                        f"[pre_pull_docker_images] Unhandled exception for {image}: {exc}"
                    )
                    results.append((image, False, str(exc)))
                    failed_images.append((image, str(exc)))
                finally:
                    pbar.update(1)

    # Summary
    success_count = sum(1 for r in results if r[1])
    failure_count = len(results) - success_count

    main_logger.info(
        f"[pre_pull_docker_images] Pre-pull completed: {success_count} succeeded, {failure_count} failed."
    )

    if failed_images:
        main_logger.info(
            "[pre_pull_docker_images] Failed to pull the following images:"
        )
        for img, err in failed_images:
            main_logger.info(f"- {img}: {err}")


def collect_rows(
    repo_name, docker_images, max_workers=100, debug=False, collect_timeout=400
):
    main_logger.info(
        f"Collecting rows for {repo_name} with {len(docker_images)} images"
    )
    all_rows: list[DatasetRow] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(
                validate_docker_and_get_row, (repo_name, docker_image, True)
            ): docker_image
            for docker_image in docker_images
        }

        with tqdm.tqdm(total=len(future_to_image), desc="Collecting Rows") as pbar:
            for future in as_completed(future_to_image):
                docker_image = future_to_image[future]
                try:
                    row = future.result(timeout=collect_timeout)
                except TimeoutError:
                    main_logger.info(
                        f"[collect_rows] Timeout collecting row for {docker_image}"
                    )
                    row = None
                except Exception as e:
                    main_logger.info(
                        f"[collect_rows] Failed to collect row for {docker_image}: {e}"
                    )
                    if debug:
                        main_logger.info(
                            f"Failed to collect row for {docker_image}: {e}"
                        )
                        import traceback

                        traceback.print_exc()
                    row = None

                if row:
                    all_rows.append(row)
                    main_logger.info(
                        f"[collect_rows] Successfully collected row for {docker_image}"
                    )
                else:
                    main_logger.info(f"[collect_rows] No row data for {docker_image}")
                pbar.update(1)
    return all_rows


def main():

    repo_names = [
        "tornado",
        "pillow",
        "scrapy",
        "pyramid",
        "datalad",
        "aiohttp",
        "numpy",
        "orange3",
        #
        # "pandas",
        # "coveragepy",
    ]

    BATCH_SIZE = 50
    USE_EXISTING = True

    for repo_name in repo_names:

        if USE_EXISTING and os.path.exists(f"repo_datasets/{repo_name}.jsonl"):
            with open(f"repo_datasets/{repo_name}.jsonl", "r", encoding="utf-8") as f:
                existing_rows = [DatasetRow(**json.loads(line)) for line in f]
                existing_images = [row.docker_image for row in existing_rows]
        else:
            existing_rows = []
            existing_images = []

        main_logger.info(f"Starting to collect rows for {repo_name}")
        all_docker_images = fetch_docker_tags_for_repo(repo_name)
        if USE_EXISTING:
            all_docker_images = list(set(all_docker_images) - set(existing_images))
        num_batches = math.ceil(len(all_docker_images) / BATCH_SIZE)

        main_logger.info(
            f"Found {len(all_docker_images)} images for {repo_name}, processing in {num_batches} batches"
        )

        count = 0
        with open(f"repo_datasets/{repo_name}.jsonl", "w", encoding="utf-8") as f:
            if USE_EXISTING:
                for row in existing_rows:
                    f.write(row.model_dump_json(indent=None) + "\n")
                    count += 1
                main_logger.info(f"Re-Written {count} existing rows for {repo_name}")

            for i in range(num_batches):
                main_logger.info(f"Starting batch {i+1}/{num_batches} for {repo_name}")
                docker_images = all_docker_images[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

                pre_pull_docker_images(docker_images)

                new_rows = collect_rows(repo_name, docker_images)

                for row in new_rows:
                    if row:
                        f.write(row.model_dump_json(indent=None) + "\n")
                        count += 1
                main_logger.info(
                    f"Written {count} rows for {repo_name} after {i+1}/{num_batches} batches"
                )

                # for image in docker_images:
                #     docker.client.from_env().images.remove(image, force=True)

                # subprocess.run(
                #     ["docker", "system", "prune", "-a", "-f"],
                #     capture_output=True,
                #     check=True,
                # )


if __name__ == "__main__":
    main_logger.info("Starting validation")
    main()
