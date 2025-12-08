import os
import json
import subprocess
from pathlib import Path

from tqdm import tqdm

from r2egym.commit_models.diff_classes import (
    ParsedCommit,
    FileDiff,
    UniHunk,
    LineType,
)
from r2egym.commit_models.entity_utils import (
    EntityType,
    Entity,
    build_code_structure,
    pprint_entities,
)


class CommitAnalyzer:

    def __init__(self, parsed_commit: ParsedCommit):
        self.parsed_commit = parsed_commit

    def analyze_commit(self, verbose: bool = False):
        """
        Analyze the commit by checking out the 'before' and 'after' versions of each file
        """
        for file_diff in self.parsed_commit.file_diffs:
            if file_diff.is_python_file:
                self.analyze_file(file_diff, verbose)

    def analyze_file(self, file_diff: FileDiff, verbose: bool = False):
        """
        Analyze a single file in the commit.
        Loads the 'before' and 'after' versions of the file
        For each file, load the entities in the 'before' and 'after' versions.
        Additionally, we map the entities to the lines they occupy in the file.
        Next, we analyze each hunk in the file.
        """

        before_code = file_diff.old_file_content

        after_code = file_diff.new_file_content

        code_structure_before = build_code_structure(file_diff.path, before_code)
        code_structure_after = build_code_structure(file_diff.path, after_code)

        all_entities_before = code_structure_before.entities
        all_entities_after = code_structure_after.entities

        before_entities_by_line = code_structure_before.entities_by_line
        after_entities_by_line = code_structure_after.entities_by_line

        for hunk in file_diff.hunks:
            self.analyze_hunk(
                hunk,
                all_entities_before,
                all_entities_after,
                before_entities_by_line,
                after_entities_by_line,
            )

            if verbose:
                print(f"\nHunk: {hunk.descriptor}")
                if hunk.edit_transcends_single_location:
                    print("Note: Edits transcend multiple entities in this hunk.")

                if hunk.modified_entities:
                    print("Modified entities:")
                    pprint_entities(hunk.modified_entities)

                if hunk.added_entities:
                    print("Added entities:")
                    pprint_entities(hunk.added_entities)

                if hunk.deleted_entities:
                    print("Deleted entities:")
                    pprint_entities(hunk.deleted_entities)

    def analyze_hunk(
        self,
        hunk: UniHunk,
        all_entities_before: list[Entity],
        all_entities_after: list[Entity],
        before_entities_by_line: dict[int, set[Entity]],
        after_entities_by_line: dict[int, set[Entity]],
    ):
        """
        Analyze a single hunk in the commit.
        Determine the entities affected by the hunk.
        Uses the entities in the 'before' and 'after' versions of the file.
        Particularly, for each modified line in the hunk, we determine the entities
        that occupy that line thus identifying the entities affected by the hunk.
        We use the entities in the 'before' and 'after' versions of the file to determine
        if an entity was modified, added, or deleted.
        """

        # Lines in 'before' code (deleted or modified)
        modified_line_numbers_after = set()
        current_lineno = hunk.descriptor.new_range.start
        for line in hunk.line_group.all_lines:
            if line.type == LineType.ADDED:
                modified_line_numbers_after.add(current_lineno)
                current_lineno += 1
            elif line.type == LineType.CONTEXT:
                current_lineno += 1

        # Lines in 'after' code (added or modified)
        modified_line_numbers_before = set()
        current_lineno = hunk.descriptor.old_range.start
        for line in hunk.line_group.all_lines:
            if line.type == LineType.DELETED:
                modified_line_numbers_before.add(current_lineno)
                current_lineno += 1
            elif line.type == LineType.CONTEXT:
                current_lineno += 1

        # Entities affected in 'before' code
        affected_entities_before: set[Entity] = set()
        for lineno in modified_line_numbers_before:
            if lineno in before_entities_by_line:
                affected_entities_before.update(before_entities_by_line[lineno])

        # Entities affected in 'after' code
        affected_entities_after: set[Entity] = set()
        for lineno in modified_line_numbers_after:
            if lineno in after_entities_by_line:
                affected_entities_after.update(after_entities_by_line[lineno])

        total_affected_entities = affected_entities_before.union(
            affected_entities_after
        )

        modified_entities: set[Entity] = set()
        added_entities: set[Entity] = set()
        deleted_entities: set[Entity] = set()

        for entity in total_affected_entities:
            if entity in all_entities_before and entity in all_entities_after:
                modified_entities.add(entity)
            elif entity in all_entities_after:
                added_entities.add(entity)
            else:
                deleted_entities.add(entity)

        hunk.modified_entities = modified_entities
        hunk.added_entities = added_entities
        hunk.deleted_entities = deleted_entities

        return


if __name__ == "__main__":

    commit_paths = [
        # "commit_data/sympy/0a536f722df2ab8a24a46c6aef77f54bd66965d1.json",
        # "commit_data/sympy/fffa4d82a4d81de061843a74b790c5b52bd06676.json",
        # "commit_data/sympy/fceccc55b8be528b21570a1b158f396e7c22b88b.json",
        # "commit_data/sympy/fceddec334202fad6cabeb0b0a8bc955189bba22.json",
        # "commit_data/sympy/fb5890928ac030cf58f838dfea9fadd1f8b8a5b7.json",
        # "commit_data/sympy/faaee0900fa2d1eef138791adce0235ef8f3a05b.json",
        # "commit_data/sympy/f5190489221d0620afe2ece0ed5301dca855e390.json",
        # "commit_data/sympy/f49997b0dd5e2c0a751605b4eb9158455d77a7ad.json",
        # "commit_data/sympy/f1659ead73f3f0bf63864129529404edc9264b00.json",
        # "commit_data/sympy/5ec4afc7c4dfc7672a11f48dce7cd6fcbfc655da.json",
        # "commit_data/sympy/9028f8f683ae1e9724eb689f2e20bed455b1f18a.json",
        "commit_data/sympy/2a28a05ca5b2768f8a7fc67f68d23a67cf1962c6.json",
        # "commit_data/sympy/1b68b7c925d478428886ccf96be1f4598549beee.json",
        # "commit_data/sympy/1a74e5af16dea2c92e16e23a5faad320ad101ee6.json",
        # "commit_data/sympy/c0e7ded22e2119e336b5daaaf63567a3d16ac494.json",
    ]
    # commit_paths = os.listdir("commit_data/sympy")
    # commit_paths = [f"commit_data/sympy/{commit_path}" for commit_path in commit_paths]

    for commit_path in tqdm(commit_paths):
        with open(commit_path) as f:
            parsed_commit = ParsedCommit(**json.load(f))

        try:
            print(f"Analyzing commit {parsed_commit.new_commit_hash}")
            CommitAnalyzer(parsed_commit, Path("/tmp/sympy")).analyze_commit(True)
        except Exception as e:
            print(f"Error analyzing commit {parsed_commit.new_commit_hash}: {e}")
            continue
