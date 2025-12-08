from r2egym.commit_models.diff_classes import FileDiff, ParsedCommit
from r2egym.repo_analysis.repo_analysis_args import RepoAnalysisLoadArgs
from r2egym.commit_models.entity_utils import (
    EntityType,
    Entity,
    build_code_structure,
    unparse_entity_without_comment_docs,
)


def is_small_commit(commit: ParsedCommit, args: RepoAnalysisLoadArgs):
    return (
        (commit.num_non_test_files > 0)
        and (commit.num_hunks > 0)
        and (commit.num_non_test_edited_lines > 0)
        and (commit.num_non_test_files < args.max_num_non_test_files)
        and (commit.num_non_test_edited_lines < args.max_num_non_test_edited_lines)
        and (len(commit.get_patch()) < args.max_patch_length)
    )


def is_long_commit(commit: ParsedCommit, args: RepoAnalysisLoadArgs):
    return not is_small_commit(commit, args)


def is_non_python_commit(commit: ParsedCommit):
    return not commit.is_only_python_edit


def is_python_commit(commit: ParsedCommit):
    return commit.is_only_python_edit


def bugedit_type_commit(commit: ParsedCommit, args: RepoAnalysisLoadArgs):
    return (
        commit.num_deleted_entities(False) == args.max_num_nontest_deleted_entities
        and commit.num_added_entities(False) <= args.max_num_nontest_added_entities
        and commit.num_edited_entities(False) <= args.max_num_nontest_edited_entities
        and commit.num_statement_entities() <= args.max_num_statement_entities
        and commit.num_edited_entities(False) > 0
    )


def filediff_has_any_non_docstring_comment_change(file_diff: FileDiff):
    if file_diff.added_entities:
        return True
    if file_diff.old_file_content is None or file_diff.new_file_content is None:
        return False
    old_code_structure = build_code_structure("", file_diff.old_file_content)
    new_code_structure = build_code_structure("", file_diff.new_file_content)
    for entity in file_diff.modified_entities:
        if entity.type in [EntityType.CLASS, EntityType.FUNCTION]:
            old_entity = old_code_structure.get_entity_by_name_type(
                entity.name, entity.type
            )
            new_entity = new_code_structure.get_entity_by_name_type(
                entity.name, entity.type
            )
            old_entity_unparse = unparse_entity_without_comment_docs(old_entity)
            new_entity_unparse = unparse_entity_without_comment_docs(new_entity)
            if old_entity_unparse != new_entity_unparse:
                return True
    return False


def has_nontest_nondocstring_comment_change(
    commit: ParsedCommit, verbose: bool = False
):
    try:
        for file_diff in commit.file_diffs:
            if not file_diff.is_test_file and file_diff.is_python_file:
                if filediff_has_any_non_docstring_comment_change(file_diff):
                    return True
    except SyntaxError as e:
        print(f"SyntaxError in commit {commit.new_commit_hash}")
        return False

    if verbose:
        print("No non-docstring comment change")
        print(commit.new_commit_hash)
        print(commit.file_name_list)
        print(commit.get_patch(False))

    return False


def issue_test_added(commit: ParsedCommit):
    all_modified_entities = commit.modified_entities(True)
    all_added_entities = commit.added_entities(True)
    non_test_modified_entities = commit.modified_entities(False)
    non_test_added_entities = commit.added_entities(False)
    test_entities = (all_modified_entities.union(all_added_entities)) - (
        non_test_modified_entities.union(non_test_added_entities)
    )
    return any("issue" in e.name for e in test_entities)


def modified_entity_test_modification(commit: ParsedCommit):
    all_modified_entities = commit.modified_entities(True)
    all_added_entities = commit.added_entities(True)
    non_test_modified_entities = commit.modified_entities(False)
    non_test_added_entities = commit.added_entities(False)
    test_entities = (all_modified_entities.union(all_added_entities)) - (
        non_test_modified_entities.union(non_test_added_entities)
    )
    non_test_entities = non_test_modified_entities

    # all_non_test_mods_tested = True
    test_entity_names = [e.name.split(".")[-1] for e in test_entities]
    test_entity_contents = [e.content for e in test_entities]

    for non_test_entity in non_test_entities:
        if non_test_entity.type == EntityType.CLASS:
            continue
        non_test_entity_name = non_test_entity.name.split(".")[-1]
        if any(non_test_entity_name in n for n in test_entity_names):
            return True
        if any(non_test_entity_name + "(" in c for c in test_entity_contents):
            return True
    print(f"github.com/numpy/numpy/commit/{commit.new_commit_hash}")

    return False


def has_testmatch_edit(commit: ParsedCommit):
    return issue_test_added(commit) or modified_entity_test_modification(commit)


def has_test_entity_edit(commit: ParsedCommit):
    all_modified_entities = commit.modified_entities(True)
    all_added_entities = commit.added_entities(True)
    non_test_modified_entities = commit.modified_entities(False)
    non_test_added_entities = commit.added_entities(False)
    test_entities = (all_modified_entities.union(all_added_entities)) - (
        non_test_modified_entities.union(non_test_added_entities)
    )
    return len(test_entities) > 0


def has_mypy_test_edit(commit: ParsedCommit):
    ## any filediff is a .test file
    for file_diff in commit.file_diffs:
        if file_diff.is_mypy_test_file:
            return True
    return False
