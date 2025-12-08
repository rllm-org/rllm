import os
import json
import math
import shutil
import traceback
import subprocess
from pathlib import Path
from collections import Counter
from multiprocessing import Pool

import tqdm
import fire

from r2e.paths import REPOS_DIR
from r2egym.bash_utils import run_subprocess_shell
from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.repo_analysis.load_repo_commits import load_commits
from r2egym.repo_analysis.build_syn_issue import build_syn_issue
from r2egym.repo_analysis.repo_testheuristics import repo_heuristics
from r2egym.repo_analysis.repo_analysis_args import RepoAnalysisTestExtractArgs
from r2egym.repo_analysis.execution_result_analysis import (
    ExecutionResult,
    CommitExecutionType,
)


def create_tests(
    temp_repo_dir: Path, test_codes: list[str], test_code_files: list[str]
):
    Path(temp_repo_dir / "r2e_tests").mkdir(exist_ok=True)

    with open(temp_repo_dir / "r2e_tests" / "__init__.py", "w") as fp:
        fp.write("")

    for test_code, test_file_name in zip(test_codes, test_code_files):
        test_file_path = temp_repo_dir / "r2e_tests" / test_file_name
        with open(test_file_path, "w") as f:
            f.write(test_code)

    ## HEURISTIC: Pillow has a helper module which is not copied so manaully copying it
    # IDEALLY (TODO:): do dependency or import level slicing but that would be slower (and annoying)

    helper_path = temp_repo_dir / "Tests" / "helper.py"

    misc_files: list[Path] = [helper_path]
    for misc_file in misc_files:
        if misc_file.exists():
            new_file = temp_repo_dir / "r2e_tests" / os.path.basename(misc_file)
            with open(misc_file) as rfp, open(new_file, "w") as wfp:
                wfp.write(rfp.read())


def setup_venv(temp_repo_dir: Path):
    res = run_subprocess_shell(
        "bash install.sh",
        cwd=temp_repo_dir,
        timeout=1200,
    )
    return res


def run_test(temp_repo_dir: Path, args: RepoAnalysisTestExtractArgs):
    res = run_subprocess_shell(
        "bash run_tests.sh",
        cwd=temp_repo_dir,
        timeout=120,
    )

    return res


def test_extractor(fn_args):
    commit: ParsedCommit
    args: RepoAnalysisTestExtractArgs
    commit, args = fn_args
    new_repo_dir = REPOS_DIR / f"{args.repo_name.value}_{commit.new_commit_hash}"
    if not args.clean_old_runs and os.path.exists(
        new_repo_dir / "execution_result.json"
    ):
        with open(new_repo_dir / "execution_result.json") as f:
            return ExecutionResult(**json.load(f))
    subprocess.run(
        ["rm", "-rf", new_repo_dir.as_posix()],
    )
    subprocess.run(
        [
            "git",
            "clone",
            "--recursive",
            args.repo_dir.as_posix(),
            new_repo_dir.as_posix(),
        ],
        capture_output=True,
        check=True,
    )
    try:
        subprocess.run(
            ["git", "checkout", commit.new_commit_hash],
            cwd=new_repo_dir,
            capture_output=True,
            check=True,
        )
    except:
        print(f"Failed to checkout {commit.new_commit_hash}")
        return ExecutionResult(
            repo_name=args.repo_name.value,
            new_commit_hash=commit.new_commit_hash,
            test_file_codes=[],
            test_file_names=[],
            setup_res_code=-1000,
            setup_res_stdout="",
            setup_res_stderr="Failed to checkout",
        )
    try:
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=new_repo_dir,
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        print("Submodule update timed out")
        pass

    install_file = f"src/r2egym/install_utils/{args.repo_name.value}_install.sh"
    shutil.copy(install_file, new_repo_dir / "install.sh")

    try:
        repo_heuristics(args.repo_name.value, new_repo_dir)

        all_edited_entities = commit.edited_entities(allow_test_file=True)
        non_test_edit_entities = commit.edited_entities(allow_test_file=False)
        test_edit_entities = all_edited_entities - non_test_edit_entities

        test_file_codes: list[str] = []
        test_file_names: list[str] = []
        old_file_names: list[str] = []
        seen_file_paths = set()
        for entity in test_edit_entities:
            file_path = new_repo_dir / entity.file_name
            assert file_path.exists()
            if file_path in seen_file_paths:
                continue
            seen_file_paths.add(file_path)
            with open(file_path, "r") as fp:
                file_content = fp.read()

            test_file_codes.append(file_content)
            test_file_names.append(f"test_{len(seen_file_paths)}.py")
            old_file_names.append(entity.file_name)

        ## TODO: hack since pillow uses unittests with custom nose something :\
        ## TODO: move to heuristics above
        if args.repo_name.value == "pillow":
            if any(["unittest" in test_code for test_code in test_file_codes]):
                test_cmd = (
                    ".venv/bin/python -W ignore r2e_tests/unittest_custom_runner.py"
                )
                test_file_names.append("unittest_custom_runner.py")
                with open("src/r2egym/install_utils/unittest_custom_runner.py") as fp:
                    test_file_codes.append(fp.read())
            else:
                test_cmd = args.tests_cmd
        elif args.repo_name.value == "tornado":
            test_cmd = ".venv/bin/python -W ignore r2e_tests/tornado_unittest_runner.py"
            test_file_names.append("tornado_unittest_runner.py")
            with open("src/r2egym/install_utils/tornado_unittest_runner.py") as fp:
                test_file_codes.append(fp.read())

        elif args.repo_name.value == "datalad":
            with open("src/r2egym/install_utils/datalads_conftest.py") as fp:
                test_file_codes.append(fp.read())
                test_file_names.append("conftest.py")
                old_file_names.append("a/b/c/conftest.py")
            for idx in range(len(test_file_codes)):
                old_file_name_split = str(old_file_names[idx]).split("/")

                old_file_name_dot = ".".join(old_file_name_split[:-1]) + "."
                old_file_name_dot_dot = ".".join(old_file_name_split[:-2]) + "."
                old_file_name_dot_dot_dot = ".".join(old_file_name_split[:-3]) + "."

                test_file_codes[idx] = (
                    test_file_codes[idx]
                    .replace("from ...", f"from {old_file_name_dot_dot_dot}")
                    .replace("from ..", f"from {old_file_name_dot_dot}")
                    .replace("from .", f"from {old_file_name_dot}")
                )

            test_cmd = args.tests_cmd
        elif args.repo_name.value == "aiohttp":
            conftest_path = new_repo_dir / "tests" / "conftest.py"
            if conftest_path.exists():
                with open(conftest_path) as fp:
                    test_file_codes.append(fp.read())
                    test_file_names.append("conftest.py")
                    old_file_names.append("tests/conftest.py")

            with open(
                "src/r2egym/install_utils/process_aiohttp_updateasyncio.py"
            ) as fp:
                with open(new_repo_dir / "process_aiohttp_updateasyncio.py", "w") as f:
                    f.write(fp.read())

            test_cmd = args.tests_cmd
        elif args.repo_name.value == "numpy":
            test_cmd = args.tests_cmd
            for idx in range(len(test_file_codes)):
                ## old nosetest stuff replace
                test_file_codes[idx] = (
                    test_file_codes[idx]
                    .replace("def setup(self)", "def setup_method(self)")
                    .replace("def teardown(self)", "def teardown_method(self)")
                )

                old_file_name_split = str(old_file_names[idx]).split("/")

                old_file_name_dot = ".".join(old_file_name_split[:-1]) + "."
                old_file_name_dot_dot = ".".join(old_file_name_split[:-2]) + "."
                old_file_name_dot_dot_dot = ".".join(old_file_name_split[:-3]) + "."

                test_file_codes[idx] = (
                    test_file_codes[idx]
                    .replace("from ...", f"from {old_file_name_dot_dot_dot}")
                    .replace("from ..", f"from {old_file_name_dot_dot}")
                    .replace("from .", f"from {old_file_name_dot}")
                )

        else:
            test_cmd = args.tests_cmd

        with open(new_repo_dir / "run_tests.sh", "w") as f:
            f.write(test_cmd)

        create_tests(new_repo_dir, test_file_codes, test_file_names)
        setup_res = setup_venv(new_repo_dir)
        result = ExecutionResult(
            repo_name=args.repo_name.value,
            new_commit_hash=commit.new_commit_hash,
            test_file_codes=test_file_codes,
            test_file_names=test_file_names,
            setup_res_code=setup_res.returncode,
            setup_res_stdout=setup_res.stdout,
            setup_res_stderr=setup_res.stderr,
        )
        if setup_res.returncode != 0:
            return result

        new_test_res = run_test(new_repo_dir, args)

        result = ExecutionResult(
            repo_name=args.repo_name.value,
            new_commit_hash=commit.new_commit_hash,
            test_file_codes=test_file_codes,
            test_file_names=test_file_names,
            setup_res_code=setup_res.returncode,
            setup_res_stdout=setup_res.stdout,
            setup_res_stderr=setup_res.stderr,
            new_commit_res_code=new_test_res.returncode,
            new_commit_res_stdout=new_test_res.stdout,
            new_commit_res_stderr=new_test_res.stderr,
        )

        subprocess.run(
            ["git", "checkout", commit.old_commit_hash],
            cwd=new_repo_dir,
            capture_output=True,
            check=True,
        )

        old_test_res = run_test(new_repo_dir, args)

        result = ExecutionResult(
            repo_name=args.repo_name.value,
            new_commit_hash=commit.new_commit_hash,
            test_file_codes=test_file_codes,
            test_file_names=test_file_names,
            setup_res_code=setup_res.returncode,
            setup_res_stdout=setup_res.stdout,
            setup_res_stderr=setup_res.stderr,
            new_commit_res_code=new_test_res.returncode,
            new_commit_res_stdout=new_test_res.stdout,
            new_commit_res_stderr=new_test_res.stderr,
            old_commit_res_code=old_test_res.returncode,
            old_commit_res_stdout=old_test_res.stdout,
            old_commit_res_stderr=old_test_res.stderr,
        )

        return result
    except Exception as e:
        print(f"Error in commit {commit.new_commit_hash}: {e}")
        return ExecutionResult(
            repo_name=args.repo_name.value,
            new_commit_hash=commit.new_commit_hash,
            test_file_codes=[],
            test_file_names=[],
            setup_res_code=-1000,
            setup_res_stdout="",
            setup_res_stderr=traceback.format_exc(),
        )


def dockerize_commit_result(
    arg: tuple[ParsedCommit, ExecutionResult, RepoAnalysisTestExtractArgs],
):
    commit, result, args = arg
    commit_execution_type, improved_fn_list = result.is_good_exec()

    image_name = None

    with open(result.new_repo_dir / "execution_result.json", "w") as f:
        f.write(result.model_dump_json(indent=4))
    with open(result.new_repo_dir / "parsed_commit.json", "w") as f:
        f.write(commit.model_dump_json(indent=4))
    modified_files = commit.file_name_list
    with open(result.new_repo_dir / "modified_files.json", "w") as f:
        json.dump(modified_files, f, indent=4)
    modified_entities = commit.edited_entities()
    with open(result.new_repo_dir / "modified_entities.json", "w") as f:
        json.dump(
            [entity.json_summary_dict() for entity in modified_entities], f, indent=4
        )
    try:
        if commit_execution_type == CommitExecutionType.NEW_COMMIT_BETTER:

            ## check if docker image already exists on dockerhub
            ## TODO: add cmdline flag to skip this
            res = subprocess.run(
                [
                    "docker",
                    "pull",
                    f"namanjain12/{args.repo_name.value}_final:{commit.new_commit_hash}",
                ],
                capture_output=True,
            )
            if not args.rebuild_dockers and res.returncode == 0:
                print(f"Image already exists for {commit.new_commit_hash}")
                image_name = (
                    f"namanjain12/{args.repo_name.value}_final:{commit.new_commit_hash}"
                )
            else:
                # find . -name '*.pyc' -delete
                subprocess.run(
                    ["find", ".", "-name", "*.pyc", "-delete"],
                    cwd=result.new_repo_dir,
                    capture_output=True,
                    check=True,
                )
                # find . -name '__pycache__' -exec rm -rf {} +
                subprocess.run(
                    [
                        "find",
                        ".",
                        "-name",
                        "__pycache__",
                        "-exec",
                        "rm",
                        "-rf",
                        "{}",
                        "+",
                    ],
                    cwd=result.new_repo_dir,
                    capture_output=True,
                    check=True,
                )
                prompt, model_output, syn_issue = build_syn_issue(commit, result, args)
                if model_output == "-1":
                    return commit_execution_type, image_name
                print(syn_issue)

                with open(result.new_repo_dir / "syn_issue.json", "w") as f:
                    dump = {
                        "model_output": model_output,
                        "syn_issue": syn_issue,
                        "prompt": prompt,
                    }
                    json.dump(dump, f, indent=4)

                with open(args.parameterized_dockerfile) as f:
                    dockerfile_content = f.read()

                with open(result.new_repo_dir / "Dockerfile", "w") as f:
                    f.write(dockerfile_content)

                list_files_path = Path("tests/gym_environment/env2_utils/list_files.py")
                read_file_path = Path("tests/gym_environment/env2_utils/read_file.py")

                # with open(result.new_repo_dir / "list_files.py", "w") as f:
                #     f.write(list_files_path.read_text())

                # with open(result.new_repo_dir / "read_file.py", "w") as f:
                #     f.write(read_file_path.read_text())

                with open(result.new_repo_dir / "expected_test_output.json", "w") as f:
                    json.dump(result.new_commit_log_parse, f, indent=4)

                if args.build_dockers:

                    print(f"Building docker image for {commit.new_commit_hash}")
                    res = run_subprocess_shell(
                        f'docker build --memory 1000000000  -t namanjain12/{args.repo_name.value}_final:{commit.new_commit_hash} . --build-arg OLD_COMMIT="{commit.old_commit_hash}"',
                        capture_output=True,
                        cwd=result.new_repo_dir,
                        timeout=1200,
                    )
                    if res.returncode != 0:
                        print(res.stdout)
                        print(res.stderr)
                    else:
                        if args.push_dockers:
                            print(f"Pushing docker image for {commit.new_commit_hash}")
                            subprocess.run(
                                [
                                    "docker",
                                    "push",
                                    f"namanjain12/{args.repo_name.value}_final:{commit.new_commit_hash}",
                                ],
                                capture_output=True,
                                check=True,
                            )
                            image_name = f"namanjain12/{args.repo_name.value}_final:{commit.new_commit_hash}"

                with open(
                    f"{args.test_data_dir}/{result.new_commit_hash}.json", "w"
                ) as f:
                    f.write(result.model_dump_json(indent=4))
        return commit_execution_type, image_name
    except Exception as e:
        print(f"Error in dockerizing commit {commit.new_commit_hash}: {e}")
        import traceback

        traceback.print_exc()
        return commit_execution_type, image_name


def main(args: RepoAnalysisTestExtractArgs):
    commits = load_commits(args)
    stats = Counter()
    image_names: list[str] = []

    all_args = [(commit, args) for commit in commits]

    num_chunks = math.ceil(len(all_args) / args.chunk_size)

    for i in range(args.start_chunk, num_chunks):
        print(f"Processing chunk {i + 1}/{num_chunks}")
        sub_args = all_args[i * args.chunk_size : (i + 1) * args.chunk_size]
        sub_commits = commits[i * args.chunk_size : (i + 1) * args.chunk_size]
        with Pool(args.n_cpus) as p:
            results = list(
                tqdm.tqdm(p.imap(test_extractor, sub_args), total=len(sub_args))
            )

        arg_list = list(zip(sub_commits, results, [args] * len(sub_commits)))

        with Pool(args.n_cpus_docker) as p:
            results = list(
                tqdm.tqdm(
                    p.imap(dockerize_commit_result, arg_list), total=len(arg_list)
                )
            )
        for res in results:
            exec_type, image_name = res
            stats[exec_type] += 1

            if image_name is not None:
                image_names.append(image_name)

        print(stats)
        print(image_names)

        ## cleanup dockers for space
        if args.cleanup_dockers:
            # docker system prune -a
            subprocess.run(
                ["docker", "system", "prune", "-a", "-f"],
                capture_output=True,
                check=True,
            )


if __name__ == "__main__":
    repo_analysis_args: RepoAnalysisTestExtractArgs = fire.Fire(
        RepoAnalysisTestExtractArgs
    )
    main(repo_analysis_args)
