import datetime

import fire
import pandas as pd

from r2egym.repo_analysis.load_repo_commits import load_commits, RepoAnalysisLoadArgs

## scratch code for testing the filters

if __name__ == "__main__":
    repo_analysis_args: RepoAnalysisLoadArgs = fire.Fire(RepoAnalysisLoadArgs)
    commits = load_commits(repo_analysis_args)
    stats = []
    dates = []
    for commit_idx, commit in enumerate(commits):
        non_test_modified_entities = commit.modified_entities(False)
        non_test_added_entities = commit.added_entities(False)
        all_modified_entities = commit.modified_entities(True)
        all_added_entities = commit.added_entities(True)
        non_test_entities = non_test_modified_entities
        test_entities = (all_modified_entities.union(all_added_entities)) - (
            non_test_modified_entities.union(non_test_added_entities)
        )
        stats.append(
            {
                "commit": commit.new_commit_hash,
                "num_non_test_entities": len(non_test_entities),
                "num_test_entities": len(test_entities),
                "num_edited_lines": commit.num_edited_lines,
                "num_non_test_edited_lines": commit.num_non_test_edited_lines,
                "num_files": commit.num_files,
                "num_non_test_files": commit.num_non_test_files,
            }
        )
        dates.append(commit.commit_date)  # type datetime.datetime

    stats_df = pd.DataFrame(stats)
    print(stats_df)
    print(stats_df.describe(percentiles=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]))

    dates_df = pd.DataFrame(dates)
    print(dates_df)
    print(dates_df.describe(percentiles=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]))

    # after july 2019
    ## NOTE type
    import pytz

    print(
        len(
            [
                date
                for date in dates
                if date > datetime.datetime(2016, 1, 1).replace(tzinfo=pytz.UTC)
            ]
        )
    )
