import re
import subprocess
from pathlib import Path
from datetime import datetime

from r2egym.commit_models.diff_classes import (
    ParsedCommit,
    FileDiff,
    FileDiffHeader,
    IndexLine,
    FileInfo,
    UniHunk,
    UnitHunkDescriptor,
    Range,
    LineGroup,
    Line,
    LineType,
)


class CommitParser:
    def parse_git_diff(
        self,
        old_commit_hash: str,
        new_commit_hash: str,
        diff_text: str,
        commit_message: str,
        commit_date: datetime,
        repo_location: Path | None,
    ) -> ParsedCommit:
        file_diffs = []
        current_file_diff: FileDiff | None = None
        current_hunk: UniHunk | None = None

        for line in diff_text.split("\n"):
            if line.startswith("diff --git"):
                if current_file_diff:
                    file_diffs.append(current_file_diff)
                current_file_diff = self.parse_file_diff_header(
                    line, old_commit_hash, new_commit_hash, repo_location
                )
                current_hunk = None
            elif current_file_diff:
                if line.startswith("@@ "):
                    current_hunk = self.parse_hunk_header(line)
                    current_file_diff.hunks.append(current_hunk)
                if current_hunk is None:
                    if line.startswith("index"):
                        self.parse_file_diff_content(
                            file_diff=current_file_diff, line=line
                        )
                    elif any(
                        [
                            line.startswith(mode_prefix)
                            for mode_prefix in [
                                "old mode",
                                "new mode",
                                "deleted file mode",
                                "new file mode",
                            ]
                        ]
                    ):
                        if current_file_diff.header.misc_line:
                            current_file_diff.header.misc_line += f"\n{line}"
                        else:
                            current_file_diff.header.misc_line = line

                    elif line.startswith("+++ "):
                        current_file_diff.plus_file = FileInfo(path=line[4:])
                    elif line.startswith("--- "):
                        current_file_diff.minus_file = FileInfo(path=line[4:])
                    elif line.startswith("Binary files"):
                        current_file_diff.is_binary_file = True
                        current_file_diff.binary_line = line
                    elif line == "":
                        continue
                    else:
                        raise ValueError(f"Unexpected line: {line}")
                else:
                    self.parse_hunk_line(current_hunk, line)

        if current_file_diff:
            file_diffs.append(current_file_diff)

        return ParsedCommit(
            file_diffs=file_diffs,
            old_commit_hash=old_commit_hash,
            new_commit_hash=new_commit_hash,
            commit_message=commit_message,
            commit_date=commit_date,
        )

    def parse_file_diff_header(
        self,
        header: str,
        old_commit_hash: str,
        new_commit_hash: str,
        repo_path: Path | None,
    ) -> FileDiff:
        match = re.match(
            r'diff --git (?:")?a/([^"]+)(?:")? (?:")?b/([^"]+)(?:")?',
            header,
            re.UNICODE,
        )
        if not match:
            raise ValueError(f"Invalid diff header: {header}")

        old_path, new_path = match.groups()
        assert old_path and new_path, f"Invalid paths: {old_path}, {new_path}"
        assert (
            old_path == new_path
        ), f"Invalid paths: {old_path}, {new_path} ; usually means file was renamed which is not supported"

        if repo_path is None:
            return FileDiff(
                old_file_content="",
                new_file_content="",
                header=FileDiffHeader(file=FileInfo(path=old_path)),
            )

        old_file_content = subprocess.run(
            ["git", "--no-pager", "show", f"{old_commit_hash}:{old_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        ).stdout
        new_file_content = subprocess.run(
            ["git", "--no-pager", "show", f"{new_commit_hash}:{new_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        ).stdout

        header = FileDiffHeader(
            file=FileInfo(path=old_path),
        )
        return FileDiff(
            old_file_content=old_file_content,
            new_file_content=new_file_content,
            header=header,
        )

    def parse_file_diff_content(self, file_diff: FileDiff, line: str):
        if line.startswith("index"):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid index line: {line}")
            assert parts[0] == "index"
            hashes = parts[1].split("..")
            if len(hashes) != 2:
                raise ValueError(f"Invalid index hashes: {parts[1]}")
            file_diff.index_line = IndexLine(
                old_commit_hash=hashes[0],
                new_commit_hash=hashes[1],
                mode=parts[2] if len(parts) > 2 else "",
            )

    def parse_hunk_header(self, header: str) -> UniHunk:
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", header)
        if not match:
            raise ValueError(f"Invalid hunk header: {header}")

        old_start, old_length, new_start, new_length, section = match.groups()
        return UniHunk(
            descriptor=UnitHunkDescriptor(
                old_range=Range(
                    start=int(old_start), length=int(old_length) if old_length else None
                ),
                new_range=Range(
                    start=int(new_start), length=int(new_length) if new_length else None
                ),
                section=section.lstrip(),
            ),
            line_group=LineGroup(),
        )

    def parse_hunk_line(self, hunk: UniHunk, line: str) -> str | None:
        """
        Parses a single line within a hunk and updates the LineGroup accordingly.
        Returns the updated state.
        """
        if line == "" or line.startswith(" "):
            context_line = Line(content=line[1:], type=LineType.CONTEXT)
            hunk.line_group.all_lines.append(context_line)
        elif line.startswith("-"):
            left_line = Line(content=line[1:], type=LineType.DELETED)
            hunk.line_group.all_lines.append(left_line)
        elif line.startswith("+"):
            right_line = Line(content=line[1:], type=LineType.ADDED)
            hunk.line_group.all_lines.append(right_line)
        elif line.startswith("\\"):
            # Note lines are typically for things like "No newline at end of file"
            note_line = Line(content=line[2:], type=LineType.NOTE)
            hunk.line_group.all_lines.append(note_line)

        return

    def parse_commit(
        self,
        old_commit_hash: str,
        new_commit_hash: str,
        diff_text: str,
        commit_message: str,
        commit_date: datetime,
        repo_location: Path | None,
    ) -> ParsedCommit:
        """
        Parse a diff message and return a ParsedCommit object
        """
        return self.parse_git_diff(
            old_commit_hash,
            new_commit_hash,
            diff_text,
            commit_message,
            commit_date,
            repo_location,
        )
