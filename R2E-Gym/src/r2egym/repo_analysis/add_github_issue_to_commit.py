import re
import os
import json
import subprocess
from time import sleep
from concurrent.futures import ThreadPoolExecutor

import tqdm
from ghapi.all import GhApi

from datasets import load_dataset, Dataset


# https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests
PR_KEYWORDS = {
    "close",
    "closes",
    "closed",
    "fix",
    "fixes",
    "fixed",
    "resolve",
    "resolves",
    "resolved",
}

GHAPI_TOKEN = os.environ.get("GHAPI_TOKEN")
api = GhApi(token=GHAPI_TOKEN)

repo_name_map = {
    "pandas": "pandas-dev/pandas",
    "numpy": "numpy/numpy",
    "pillow": "python-pillow/Pillow",
    "orange3": "biolan/orange3",
    "datalad": "datalad/datalad",
    "coveragepy": "nedbat/coveragepy",
    "aiohttp": "aio-libs/aiohttp",
    "tornado": "tornadoweb/tornado",
    "pyramid": "Pylons/pyramid",
    "scrapy": "scrapy/scrapy",
}


def get_linked_pr(commit_message, commit_hash, repo_path):
    pr_pattern = r"\(#(\d+)\)"
    match = re.search(pr_pattern, commit_message)
    if match:
        return match.group(1)

    # Find merged pr in nearby direct ancestry path
    try:
        cmd = [
            "git",
            "log",
            "--merges",
            "--ancestry-path",
            "--pretty=%H %s",
            "--reverse",
            f"{commit_hash}^..HEAD",
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=repo_path,
        )

        merge_pattern = r"#(\d+)"
        merge_commits = result.stdout.strip().split("\n")[:10]
        for mc in merge_commits:
            if "Merge pull request" in mc:
                match = re.search(merge_pattern, mc)
                if match:
                    return match.group(1)
    except Exception as e:
        pass

    return None


def get_resolved_issues(repo_owner, repo_name, pr_num: str) -> str:
    """Get the conversation for a pull request.

    Goal: capture any information that might be related to testing the PR's performance.

    Note: this is not PR reviews, but regular comments on the PR.
    PR reviews usually don't have interesting testing information and also
    contain older code edits that are not relevant to the state after PR merge.
    """

    # use ghapi to get the PR discussion messages
    try:
        pr = api.pulls.get(repo_owner, repo_name, int(pr_num))
        # comments = api.issues.list_comments(repo_owner, repo_name, int(pr_num))
        # comments_str = format_comments(comments)
    except Exception as e:
        if "API rate limit exceeded" in str(e):
            print("API rate limit exceeded. Waiting for 60 seconds...")
            sleep(60)
            return get_resolved_issues(repo_owner, repo_name, pr_num)
        print(f"Error fetching PR {pr_num} for {repo_owner}/{repo_name}: {e}")
        return None, []

    resp = ""

    if pr:
        if pr.title and pr.title != "":
            resp += f"Title: {pr.title.strip()}\n"
        if pr.body and pr.body != "":
            resp += f"Description: {pr.body.strip()}"

    issues_pat = re.compile(r"(\w+)\s+\#(\d+)")

    references = dict(issues_pat.findall(resp))
    resolved_issues = list()
    if references:
        for word, issue_num in references.items():
            if word.lower() in PR_KEYWORDS:
                resolved_issues.append(issue_num)

    return resp, resolved_issues


def get_issues_for_row(row):
    pr_num = get_linked_pr(
        commit_message=json.loads(row["parsed_commit_content"])["commit_message"],
        commit_hash=row["commit_hash"],
        repo_path=row["repo_name"],
    )
    if pr_num is None:
        return pr_num, None, []

    pr_body, resolved_issues = get_resolved_issues(
        row["repo_owner"],
        row["repo_name"],
        pr_num,
    )
    return pr_num, pr_body, resolved_issues


def main():
    ds = load_dataset("r2e-edits/r2e-dockers-v1")["train"].to_pandas()
    ds["repo_owner"] = ds["repo_name"].apply(lambda x: repo_name_map[x].split("/")[0])
    ds["repo_name"] = ds["repo_name"].apply(lambda x: repo_name_map[x].split("/")[1])
    ds["pr_num"] = [None] * len(ds)
    ds["pr_body"] = [None] * len(ds)
    ds["resolved_issues"] = [None] * len(ds)

    rows = ds.iterrows()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(get_issues_for_row, row): idx for idx, row in rows}

        for future in tqdm.tqdm(
            futures,
            desc="Processing rows",
            total=len(futures),
            unit="row",
        ):
            idx = futures[future]
            try:
                pr_num, pr_body, resolved_issues = future.result()
                ds.at[idx, "pr_num"] = pr_num
                ds.at[idx, "pr_body"] = pr_body
                ds.at[idx, "resolved_issues"] = resolved_issues
            except Exception as e:
                print(f"Error processing row {ds.iloc[idx]['commit_hash']}: {e}")

    print("Done")
    print(ds["pr_num"])
    print(ds["pr_body"])
    print(ds["resolved_issues"])

    # non nan counts
    print("PR num counts:")
    print(ds["pr_num"].notna().sum())
    print("PR body counts:")
    print(ds["pr_body"].notna().sum())
    print("Resolved issues counts:")
    print(ds["resolved_issues"].notna().sum())
    print("Resolved issues counts:")
    print(
        ds["resolved_issues"]
        .apply(lambda x: len(x) if isinstance(x, list) else 0)
        .sum()
    )
    print("Resolved issues counts:")
    print(
        ds["resolved_issues"]
        .apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        .sum()
    )

    dataset = Dataset.from_pandas(ds)
    dataset.push_to_hub(
        "r2e-edits/r2e-dockers-v2",
        split="train",
    )

    pass


if __name__ == "__main__":
    main()
