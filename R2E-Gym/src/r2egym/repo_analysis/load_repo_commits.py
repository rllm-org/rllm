import os
import json
import pytz
import datetime
from typing import Callable
from multiprocessing import Pool

import tqdm
import fire

from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.repo_analysis.commit_data_heuristics import (
    is_small_commit,
    is_python_commit,
    has_nontest_nondocstring_comment_change,
    bugedit_type_commit,
    has_testmatch_edit,
    has_test_entity_edit,
    has_mypy_test_edit,
)
from r2egym.repo_analysis.repo_analysis_args import RepoAnalysisLoadArgs


def filter_fn(
    commit_datas: list[ParsedCommit],
    custom_filter_fn: Callable[[ParsedCommit], bool],
    load_run_parallel: bool = False,
    n_cpus: int = 1,
):
    if load_run_parallel:
        with Pool(processes=n_cpus) as pool:
            filter_results = list(
                tqdm.tqdm(
                    pool.imap(custom_filter_fn, commit_datas),
                    total=len(commit_datas),
                )
            )
    else:
        filter_results = [
            custom_filter_fn(
                commit_data,
            )
            for commit_data in tqdm.tqdm(commit_datas)
        ]
    commit_datas = [
        commit_data
        for commit_data, filter_result in tqdm.tqdm(
            list(zip(commit_datas, filter_results))
        )
        if filter_result
    ]
    return commit_datas


def load_commit_from_file(commit_file: str) -> ParsedCommit:
    with open(commit_file, "r") as f:
        commit_data = json.load(f)
        return ParsedCommit(**commit_data)


def load_commits_from_files(
    commit_files: list[str], load_run_parallel: bool = False, n_cpus: int = 1
) -> list[ParsedCommit]:
    if load_run_parallel:
        with Pool(processes=n_cpus) as pool:
            commit_datas = list(
                tqdm.tqdm(
                    pool.imap(
                        load_commit_from_file,
                        commit_files,
                    ),
                    total=len(commit_files),
                )
            )
    else:
        commit_datas: list[ParsedCommit] = []
        for commit_file in tqdm.tqdm(commit_files):
            with open(commit_file, "r") as f:
                commit_data = json.load(f)
                commit_datas.append(ParsedCommit(**commit_data))

    return commit_datas


def load_commits(repo_analysis_args: RepoAnalysisLoadArgs):
    commit_data_dir = repo_analysis_args.commit_data_dir

    # commit_files = ["fef3ceb2c02ef241a508eenn020fe2617e10e33e42.json"]
    commit_files = os.listdir(commit_data_dir)
    commit_files = sorted(commit_files)
    commit_files = commit_files[: repo_analysis_args.N]
    commit_files = [
        os.path.join(commit_data_dir, commit_file) for commit_file in commit_files
    ]

    commit_datas = load_commits_from_files(
        commit_files,  # repo_analysis_args.load_run_parallel, repo_analysis_args.n_cpus
    )

    if repo_analysis_args.keep_pandas_year_cutoff:
        after_july_2019_fn = (
            lambda commit_data: commit_data.commit_date
            > datetime.datetime(2016, 1, 1).replace(tzinfo=pytz.UTC)
        )
        commit_datas = filter_fn(commit_datas, after_july_2019_fn)
        if repo_analysis_args.load_verbose:
            print(f"Kept {len(commit_datas)} commits after keep_pandas_year_cutoff")

    if repo_analysis_args.load_verbose:
        print(f"Loaded {len(commit_datas)} commits")

    if repo_analysis_args.keep_only_small_commits:
        is_small_commit_parameterized = lambda commit_data: is_small_commit(
            commit_data, repo_analysis_args
        )
        commit_datas = filter_fn(commit_datas, is_small_commit_parameterized)
        if repo_analysis_args.load_verbose:
            print(
                f"Kept {len(commit_datas)} small commits after keep_only_small_commits"
            )

    if repo_analysis_args.keep_only_python_commits:
        commit_datas = filter_fn(commit_datas, is_python_commit)
        if repo_analysis_args.load_verbose:
            print(
                f"Kept {len(commit_datas)} non-python commits after keep_only_python_commits"
            )

    if repo_analysis_args.keep_only_non_docstring_commits:
        commit_datas = filter_fn(
            commit_datas,
            has_nontest_nondocstring_comment_change,
            load_run_parallel=repo_analysis_args.load_run_parallel,
            n_cpus=repo_analysis_args.n_cpus,
        )
        if repo_analysis_args.load_verbose:
            print(
                f"Kept {len(commit_datas)} non-docstring commits after keep_only_non_docstring_commits"
            )

    if repo_analysis_args.keep_only_bug_edit_commits:
        bugedit_type_commit_parameterized = lambda commit_data: bugedit_type_commit(
            commit_data, repo_analysis_args
        )
        commit_datas = filter_fn(
            commit_datas,
            bugedit_type_commit_parameterized,
        )
        if repo_analysis_args.load_verbose:
            print(
                f"Kept {len(commit_datas)} bug edit commits after keep_only_bug_edit_commits"
            )

    if repo_analysis_args.keep_only_test_entity_edit_commits:
        commit_datas = filter_fn(
            commit_datas,
            has_test_entity_edit,
        )
        if repo_analysis_args.load_verbose:
            print(
                f"Kept {len(commit_datas)} test entity commits after keep_only_test_entity_edit_commits"
            )

    if repo_analysis_args.keep_only_testmatch_commits:
        commit_datas = filter_fn(
            commit_datas,
            has_testmatch_edit,
        )
        if repo_analysis_args.load_verbose:
            print(
                f"Kept {len(commit_datas)} test match commits after keep_only_testmatch_commits"
            )

    if repo_analysis_args.keep_only_mypy_test_edit:
        commit_datas = filter_fn(
            commit_datas,
            has_mypy_test_edit,
        )
        if repo_analysis_args.load_verbose:
            print(
                f"Kept {len(commit_datas)} mypy test edit commits after keep_only_mypy_test_edit"
            )

    return commit_datas


if __name__ == "__main__":
    repo_analysis_args: RepoAnalysisLoadArgs = fire.Fire(RepoAnalysisLoadArgs)
    load_commits(repo_analysis_args)
