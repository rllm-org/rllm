"""
Pulled from official SWE-Smith repository.
"""
import os
import re
from pathlib import Path
from unidiff import PatchSet

from r2egym.swesmith.constants import (
    KEY_IMAGE_NAME,
    KEY_MIN_TESTING,
    KEY_PATCH,
    KEY_TEST_CMD,
    MAP_REPO_TO_SPECS,
)

FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"
INSTANCE_REF = "instance_ref"

def get_repo_name(repo, commit) -> str:
    """
    Get the SWE-smith GitHub repository name for a repository at a specific commit.
    """
    return f"{repo.replace('/', '__')}.{commit[:8]}"

def get_test_paths(dir_path: str, ext: str = ".py") -> list[Path]:
    """
    Get all testing file paths relative to the given directory.
    """
    return [
        Path(os.path.relpath(os.path.join(root, file), dir_path))
        for root, _, files in os.walk(Path(dir_path).resolve())
        for file in files
        if (
            (
                any([x in root.split("/") for x in ["tests", "test", "specs"]])
                or file.lower().startswith("test")
                or file.rsplit(".", 1)[0].endswith("test")
            )
            and (ext is None or file.endswith(ext))
        )
    ]


def get_full_commit(repo, partial_commit) -> str:
    """
    Get the full commit hash for a repository at a specific commit.
    """
    for commit in MAP_REPO_TO_SPECS[repo]:
        if commit.startswith(partial_commit):
            return commit

    raise ValueError(f"Commit {partial_commit} not found for repository {repo}.")

def get_repo_commit_from_image_name(image_name: str) -> tuple[str, str]:
    """
    Get the repository and commit from a docker image ID.
    """
    # Parsing supports repos with '.' in their name
    image_name = image_name.split(".", 2)[-1]
    repo = image_name.rsplit(".", 1)[0].replace("__", "/")
    partial_commit = image_name.rsplit(".", 1)[-1]
    for repo_name in MAP_REPO_TO_SPECS:
        # Hack because docker image_name must be lowercase
        if repo_name.lower() == repo:
            repo = repo_name
            break
    commit = get_full_commit(repo, partial_commit)
    return repo, commit


def get_test_command_mypy(instance: dict):
    repo, commit = get_repo_commit_from_image_name(instance[KEY_IMAGE_NAME])
    pattern = r"\[case ([^\]]+)\]"
    if FAIL_TO_PASS in instance:
        fail_to_pass_files = [x.rsplit("::", 1)[-1] for x in instance[FAIL_TO_PASS]]
        if PASS_TO_PASS in instance:
            pass_to_pass_files = [x.rsplit("::", 1)[-1] for x in instance[PASS_TO_PASS]]
            all_files = list(set(fail_to_pass_files + pass_to_pass_files))
        else:
            all_files = list(set(fail_to_pass_files))
        test_keys = " or ".join(all_files)
    elif INSTANCE_REF in instance and "test_patch" in instance[INSTANCE_REF]:
        test_keys = " or ".join(
            re.findall(pattern, instance[INSTANCE_REF]["test_patch"])
        )
    return f'{MAP_REPO_TO_SPECS[repo][commit][KEY_TEST_CMD]} "{test_keys}"'

MAP_REPO_TO_TEST_CMD = {
    "python/mypy": get_test_command_mypy,
}

def get_test_command(instance: dict):
    """
    Given a repo/commit pair and a (gold) patch, return the test command to run
    """
    repo, commit = get_repo_commit_from_image_name(instance[KEY_IMAGE_NAME])
    specs = MAP_REPO_TO_SPECS[repo][commit]
    test_command = specs[KEY_TEST_CMD]

    if FAIL_TO_PASS in instance and "pytest" in specs[KEY_TEST_CMD]:
        # NOTE: Using F2P key as indicator that this is eval instance, not validation
        if repo in MAP_REPO_TO_TEST_CMD:
            return MAP_REPO_TO_TEST_CMD[repo](instance), []
        f2p_files = list(set([x.split("::", 1)[0] for x in instance[FAIL_TO_PASS]]))
        p2p_files = []
        if PASS_TO_PASS in instance:
            p2p_files = list(set([x.split("::", 1)[0] for x in instance[PASS_TO_PASS]]))
        all_files = list(set(f2p_files + p2p_files))
        test_command += f" {' '.join(all_files)}"
        return test_command, all_files

    if KEY_MIN_TESTING not in specs or KEY_PATCH not in instance:
        # If min testing is not enabled or there's no patch
        # return test command as is (usually = run whole test suite)
        return test_command, []

    # Get all testing related file paths in the repo
    test_paths = get_test_paths(get_repo_name(repo, commit))

    if (
        INSTANCE_REF in instance
        and len(instance[INSTANCE_REF]["test_patch"].strip()) > 0
    ):
        test_patch = instance[INSTANCE_REF]["test_patch"]
        # For PR Mirroring (SWE-bench style) instances,
        # if test patch is available, use that information
        if repo in MAP_REPO_TO_TEST_CMD:
            return MAP_REPO_TO_TEST_CMD[repo](instance), []
        rv = []
        for x in PatchSet(test_patch):
            for test_path in test_paths:
                if str(test_path).endswith(x.path) or str(test_path).endswith(
                    Path(x.path).name
                ):
                    rv.append(str(test_path))
        if len(rv) > 0:
            test_command += f" {' '.join(rv)}"
            return test_command, rv

    # Identify relevant test files based on the patch
    patch_paths = [Path(f.path) for f in PatchSet(instance[KEY_PATCH])]
    rv = []
    for patch_path in patch_paths:
        file_name = patch_path.name.strip(".py")
        parent_dir = patch_path.parent.name
        for test_path in test_paths:
            # Check for common test file naming conventions first
            # If found, add to list and break
            common_test_names = [
                f"test_{file_name}.py",
                f"test{file_name}.py",
                f"{file_name}_test.py",
                f"{file_name}test.py",
            ]
            if any(
                [
                    str(test_path).endswith(f"{parent_dir}/{name}")
                    or str(test_path).endswith(name)
                    for name in common_test_names
                ]
            ):
                rv.append(str(test_path))
                break
        else:
            for test_path in test_paths:
                if parent_dir == test_path.parent.name:
                    # If similar testing folder found, add to list and break
                    rv.append(str(test_path.parent))
                    break
                elif any(
                    [
                        x.format(parent_dir) == test_path.name
                        for x in ["test_{}.py", "test{}.py", "{}_test.py", "{}test.py"]
                    ]
                ):
                    rv.append(str(test_path))

    if len(rv) > 0:
        # Remove duplicates
        test_files = [x for x in rv if x.endswith(".py")]
        final = [x for x in rv if not x.endswith(".py")]
        for test_file in test_files:
            if os.path.dirname(test_file) not in final:
                final.append(test_file)
        test_command += f" {' '.join(set(final))}"

    return test_command, rv