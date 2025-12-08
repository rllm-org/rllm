import os
import datetime
import subprocess
from multiprocessing import Pool
from collections import defaultdict

import tqdm
import fire

from r2egym.commit_models.parse_diff import CommitParser
from r2egym.commit_models.commit_to_ast import CommitAnalyzer
from r2egym.repo_analysis.repo_analysis_args import RepoAnalysisArgs


def process_commit(commit: str):
    """
    Process a single commit and return the diff message, commit message, and commit date.
    Uses git diff to get the diff message and git log to get the commit message and commit date.
    """
    try:
        old_commit = f"{commit}^"
        diff_message = subprocess.check_output(
            ["git", "diff", "-p", f"{old_commit}", commit],
            cwd=repo_analysis_args.repo_dir,
        ).decode()
        commit_message = (
            subprocess.check_output(
                ["git", "log", "-1", "--pretty=%B", commit],
                cwd=repo_analysis_args.repo_dir,
            )
            .decode()
            .strip()
        )
        commit_date = (
            subprocess.check_output(
                ["git", "log", "-1", "--pretty=%cd", commit],
                cwd=repo_analysis_args.repo_dir,
            )
            .decode()
            .strip()
        )
        commit_date = datetime.datetime.strptime(commit_date, "%a %b %d %H:%M:%S %Y %z")
        return (old_commit, commit, diff_message, commit_message, commit_date)
    except subprocess.CalledProcessError:
        return None
    except Exception as e:
        if "utf-8' codec can't decode" in str(e):
            return None
        print(f"Error processing commit {commit}: {e}")
        return None


def analyze_save_commit(good_commit_commit):
    old_commit, commit, diff_message, commit_message, commit_date = good_commit_commit
    parser = CommitParser()

    try:
        parsed_diff = parser.parse_commit(
            old_commit,
            commit,
            diff_message,
            commit_message,
            commit_date,
            repo_location=repo_analysis_args.repo_dir,
        )
    except Exception as e:
        if "usually means file was renamed" in str(e):
            return
        print(f"Error parsing commit {commit}: {e}")
        return

    try:
        assert (
            parsed_diff.get_patch(only_python=False).strip() == diff_message.strip()
        ), commit
    except AssertionError:
        print(f"Error in commit {commit}")
        with open("parsed_patch.patch", "w") as f:
            f.write(parsed_diff.get_patch())
        with open("original_patch.patch", "w") as f:
            f.write(diff_message)
        return

    try:
        CommitAnalyzer(parsed_diff).analyze_commit(False)
    except SyntaxError as e:
        print(f"skipping syntax error in commit {commit}")
        return
    except Exception as e:
        print(f"Error analyzing commit {parsed_diff.new_commit_hash}: {e}")

    with open(
        repo_analysis_args.commit_data_dir / f"{parsed_diff.new_commit_hash}.json", "w"
    ) as f:
        f.write(parsed_diff.model_dump_json(indent=4))


def main():
    commit_data_dir = repo_analysis_args.commit_data_dir
    if len(os.listdir(commit_data_dir)) > 0:
        print(f"Commit data directory {commit_data_dir} is not empty... Please check.")
        return

    commits = (
        subprocess.check_output(
            ["git", "log", "--pretty=format:%H"], cwd=repo_analysis_args.repo_dir
        )
        .decode()
        .splitlines()
    )

    with Pool(processes=repo_analysis_args.n_cpus) as pool:
        results = list(
            tqdm.tqdm(pool.imap(process_commit, commits), total=len(commits))
        )

    good_commit_commit = [result for result in results if result is not None]

    # Further processing can be done here with good_diffs
    print(f"Collected {len(good_commit_commit)} good diffs.")

    with Pool(processes=32) as pool:
        list(
            tqdm.tqdm(
                pool.imap(analyze_save_commit, good_commit_commit),
                total=len(good_commit_commit),
            )
        )


if __name__ == "__main__":
    repo_analysis_args: RepoAnalysisArgs = fire.Fire(RepoAnalysisArgs)
    main()
