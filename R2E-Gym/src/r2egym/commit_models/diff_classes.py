from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field
from r2egym.commit_models.entity_utils import EntityType, Entity


class Range(BaseModel):
    start: int
    length: int | None = None

    def get_patch(self) -> str:
        if self.length is None:
            return f"{self.start}"
        return f"{self.start},{self.length}"


class UnitHunkDescriptor(BaseModel):
    old_range: Range
    new_range: Range
    section: str

    def get_patch(self) -> str:
        content = f"@@ -{self.old_range.get_patch()} +{self.new_range.get_patch()} @@"
        if self.section:
            content += f" {self.section}"
        return content


class LineType(Enum):
    CONTEXT = "context"
    ADDED = "added"
    DELETED = "deleted"
    NOTE = "note"


class Line(BaseModel):
    content: str
    type: LineType


class LineGroup(BaseModel):
    all_lines: list[Line] = Field(default_factory=list)

    @property
    def num_deleted(self) -> int:
        return sum(line.type == LineType.DELETED for line in self.all_lines)

    @property
    def num_added(self) -> int:
        return sum(line.type == LineType.ADDED for line in self.all_lines)

    @property
    def num_context(self) -> int:
        return sum(line.type == LineType.CONTEXT for line in self.all_lines)

    @property
    def lr_lines(self) -> list[Line]:
        return [
            line
            for line in self.all_lines
            if line.type in [LineType.DELETED, LineType.CONTEXT]
        ]

    @property
    def num_edited(self) -> int:
        return self.num_deleted + self.num_added


class UniHunk(BaseModel):
    descriptor: UnitHunkDescriptor
    line_group: LineGroup
    modified_entities: set[Entity] = Field(default_factory=set)
    added_entities: set[Entity] = Field(default_factory=set)
    deleted_entities: set[Entity] = Field(default_factory=set)

    @property
    def is_import_hunk(self) -> bool:
        for line in self.line_group.lr_lines:
            if len(line.content.strip()) == 0:
                continue
            if line.content.startswith("import"):
                continue
            if line.content.startswith("from ") and "import" in line.content:
                continue
            return False
        return True

    @property
    def is_insert_hunk(self) -> bool:
        return self.line_group.num_deleted == 0

    @property
    def is_delete_hunk(self) -> bool:
        return self.line_group.num_added == 0

    @property
    def edited_entities(self) -> set[Entity]:
        return self.modified_entities.union(self.added_entities).union(
            self.deleted_entities
        )

    @property
    def num_edited_entities(self) -> int:
        return len(self.edited_entities)

    @property
    def num_modified_entities(self) -> int:
        return len(self.modified_entities)

    @property
    def num_added_entities(self) -> int:
        return len(self.added_entities)

    @property
    def num_deleted_entities(self) -> int:
        return len(self.deleted_entities)

    @property
    def num_method_entities(self) -> int:
        return sum(entity.type == EntityType.METHOD for entity in self.edited_entities)

    @property
    def num_function_entities(self) -> int:
        return sum(
            entity.type == EntityType.FUNCTION for entity in self.edited_entities
        )

    @property
    def num_class_entities(self) -> int:
        return sum(entity.type == EntityType.CLASS for entity in self.edited_entities)

    @property
    def edit_transcends_single_location(self) -> bool:
        return (self.num_function_entities + self.num_class_entities > 1) or (
            self.num_method_entities > 1
        )


class FileInfo(BaseModel):
    path: str


class FileDiffHeader(BaseModel):
    file: FileInfo
    misc_line: str | None = None

    @property
    def path(self) -> str:
        return self.file.path

    @property
    def is_test_file(self) -> bool:
        return (
            self.path.endswith("_test.py")
            or self.path.startswith("test_")
            or "tests" in self.path.split("/")
        )

    def get_patch(self) -> str:
        patch = f"diff --git a/{self.file.path} b/{self.file.path}\n"
        if self.misc_line:
            patch += self.misc_line + "\n"
        return patch


class IndexLine(BaseModel):
    old_commit_hash: str
    new_commit_hash: str
    mode: str

    def get_patch(self) -> str:
        return f"index {self.old_commit_hash}..{self.new_commit_hash}{' ' if self.mode else ''}{self.mode}\n"


class FileDiff(BaseModel):
    old_file_content: str
    new_file_content: str
    header: FileDiffHeader
    index_line: IndexLine | None = None
    is_binary_file: bool = False
    binary_line: str | None = None
    minus_file: FileInfo | None = None
    plus_file: FileInfo | None = None
    hunks: list[UniHunk] = []

    @property
    def path(self) -> str:
        return self.header.path

    @property
    def is_test_file(self) -> bool:
        return (
            self.path.endswith("_test.py")
            or self.path.startswith("test_")
            or self.path.split("/")[-1].startswith("test_")
            or "tests" in self.path.split("/")
            or "Tests" in self.path.split("/")
            or "test" in self.path.split("/")
            or "Test" in self.path.split("/")
        )

    @property
    def is_mypy_test_file(self) -> bool:
        return self.path.endswith(".test")

    def get_patch(self) -> str:
        patch = self.header.get_patch()
        if self.index_line:
            patch += self.index_line.get_patch()
        if self.is_binary_file:
            patch += self.binary_line + "\n"

        if self.minus_file and self.plus_file:
            patch += f"--- {self.minus_file.path}\n"
            patch += f"+++ {self.plus_file.path}\n"
        for hunk in self.hunks:
            patch += hunk.descriptor.get_patch() + "\n"
            for line in hunk.line_group.all_lines:
                if line.type == LineType.CONTEXT:
                    patch += f" {line.content}\n"
                elif line.type == LineType.ADDED:
                    patch += f"+{line.content}\n"
                elif line.type == LineType.DELETED:
                    patch += f"-{line.content}\n"
                elif line.type == LineType.NOTE:
                    patch += f"\\ {line.content}\n"

        return patch

    @property
    def is_python_file(self) -> bool:
        return self.path.endswith(".py")

    @property
    def num_hunks(self) -> int:
        return len(self.hunks)

    @property
    def num_edited_lines(self) -> int:
        return sum(hunk.line_group.num_edited for hunk in self.hunks)

    @property
    def edited_entities(self) -> set[Entity]:
        return {entity for hunk in self.hunks for entity in hunk.edited_entities}

    @property
    def added_entities(self) -> set[Entity]:
        return {entity for hunk in self.hunks for entity in hunk.added_entities}

    @property
    def deleted_entities(self) -> set[Entity]:
        return {entity for hunk in self.hunks for entity in hunk.deleted_entities}

    @property
    def modified_entities(self) -> set[Entity]:
        return {entity for hunk in self.hunks for entity in hunk.modified_entities}

    @property
    def num_edited_entities(self) -> int:
        return len(self.edited_entities)

    @property
    def num_added_entities(self) -> int:
        return len(self.added_entities)

    @property
    def num_deleted_entities(self) -> int:
        return len(self.deleted_entities)

    @property
    def num_modified_entities(self) -> int:
        return len(self.modified_entities)

    @property
    def num_method_entities(self) -> int:
        return sum(entity.type == EntityType.METHOD for entity in self.edited_entities)

    @property
    def num_function_entities(self) -> int:
        return sum(
            entity.type == EntityType.FUNCTION for entity in self.edited_entities
        )

    @property
    def num_class_entities(self) -> int:
        return sum(entity.type == EntityType.CLASS for entity in self.edited_entities)

    @property
    def is_new(self) -> bool:
        return self.old_file_content == "/dev/null" or self.old_file_content == ""


class ParsedCommit(BaseModel):
    """
    Represents a parsed commit, with all of its file diffs
    Contains metadata about the commit, such as the commit message and date
    """

    file_diffs: list[FileDiff]
    old_commit_hash: str
    new_commit_hash: str
    commit_message: str
    commit_date: datetime
    metadata: dict = Field(default_factory=dict)

    def get_patch(
        self,
        test_file: bool = True,
        non_test_file: bool = True,
        include_files: list | None = None,
        exclude_files=[],
        only_python=True,
    ) -> str:
        patch = ""
        for file_diff in self.file_diffs:
            if file_diff.path in exclude_files:
                continue  # Skip files in exclude_files
            if include_files and file_diff.path not in include_files:
                continue
            if only_python and not file_diff.is_python_file:
                continue
            if file_diff.is_test_file and test_file:
                patch += file_diff.get_patch()
            if not file_diff.is_test_file and non_test_file:
                patch += file_diff.get_patch()

        return patch

    @property
    def file_name_list(self) -> list[str]:
        return [file_diff.path for file_diff in self.file_diffs]

    @property
    def non_test_file_name_list(self) -> list[str]:
        return [
            file_diff.path
            for file_diff in self.file_diffs
            if not file_diff.is_test_file
        ]

    def get_file_name_list(
        self,
        test_file: bool = False,
        exclude_extensions=[
            "rst",
            "yml",
            "toml",
            "sh",
            "md",
            "txt",
            "test",
            "gitignore",
            "feature",
            "removal",
            "bugfix",
            "cfg",
            "misc",
            "doc",
            "dat",
            "ini",
            "Makefile",
            "pip",
            "treerc",
            "xml",
            "pylintrc",
            "html",
            "in",
            "ini",
            "css",
            "cfg",
        ],
    ) -> list[str]:
        return [
            file_diff.path
            for file_diff in self.file_diffs
            if (file_diff.is_test_file and test_file)
            or not file_diff.is_test_file
            and not any(file_diff.path.endswith(ext) for ext in exclude_extensions)
        ]

    @property
    def file_extension_set(self) -> set[str]:
        return {file_diff.path.split(".")[-1] for file_diff in self.file_diffs}

    @property
    def is_only_python_edit(self) -> bool:
        # if "py" in self.file_extension_set and not {
        #     "py",
        #     "rst",
        #     "yml",
        #     "toml",
        #     "sh",
        #     "md",
        #     "txt",
        #     "test",
        #     "gitignore",
        #     "feature",
        #     "removal",
        #     "bugfix",
        #     "cfg",
        #     "misc",
        #     "doc",
        #     "dat",
        #     "ini",
        #     "Makefile",
        #     "pip",
        #     "treerc",
        #     "xml",
        #     "pylintrc",
        #     "html",
        #     "in",
        #     "ini",
        #     "css",
        #     "cfg",
        # }.issuperset(self.file_extension_set):
        #     print(self.file_extension_set)
        return {
            "py",
            "rst",
            "yml",
            "toml",
            "sh",
            "md",
            "txt",
            "test",
            "gitignore",
            "feature",
            "removal",
            "bugfix",
            "cfg",
            "misc",
            "doc",
            "dat",
            "ini",
            "Makefile",
            "pip",
            "treerc",
            "xml",
            "pylintrc",
            "html",
            "in",
            "ini",
            "css",
            "cfg",
        }.issuperset(self.file_extension_set) and "py" in self.file_extension_set

    @property
    def num_files(self) -> int:
        return len(self.file_diffs)

    @property
    def num_test_files(self) -> int:
        return sum(file_diff.is_test_file for file_diff in self.file_diffs)

    @property
    def num_non_test_files(self) -> int:
        return self.num_files - self.num_test_files

    @property
    def num_hunks(self) -> int:
        return sum(len(file_diff.hunks) for file_diff in self.file_diffs)

    @property
    def num_edited_lines(self) -> int:
        return sum(file_diff.num_edited_lines for file_diff in self.file_diffs)

    def get_num_lines_edited(
        self,
        test_file: bool = True,
        non_test_file: bool = True,
        include_files: list | None = None,
        exclude_files=[],
        only_python=True,
    ):
        num_lines_edited = 0
        for file_diff in self.file_diffs:
            if file_diff.path in exclude_files:
                continue
            if include_files and file_diff.path not in include_files:
                continue
            if only_python and not file_diff.is_python_file:
                continue
            if file_diff.is_test_file and test_file:
                num_lines_edited += file_diff.num_edited_lines
            if not file_diff.is_test_file and non_test_file:
                num_lines_edited += file_diff.num_edited_lines
        return num_lines_edited

    @property
    def num_non_test_edited_lines(self) -> int:
        return sum(
            hunk.line_group.num_edited
            for file_diff in self.file_diffs
            if not file_diff.is_test_file
            for hunk in file_diff.hunks
        )

    @property
    def is_bugfix(self) -> bool:
        return (
            "fix" in self.commit_message.lower() or "bug" in self.commit_message.lower()
        )

    @property
    def is_feature(self) -> bool:
        return (
            "feature" in self.commit_message.lower()
            or "add" in self.commit_message.lower()
        )

    @property
    def is_refactor(self) -> bool:
        return "refactor" in self.commit_message.lower()

    @property
    def commit_date(self) -> datetime:
        return

    @property
    def all_hunks(self) -> list[UniHunk]:
        return [hunk for file_diff in self.file_diffs for hunk in file_diff.hunks]

    @property
    def are_all_insert_hunks(self) -> bool:
        return all(hunk.is_insert_hunk for hunk in self.all_hunks)

    @property
    def are_all_delete_hunks(self) -> bool:
        return all(hunk.is_delete_hunk for hunk in self.all_hunks)

    @property
    def are_all_import_hunks(self) -> bool:
        return all(hunk.is_import_hunk for hunk in self.all_hunks)

    @property
    def are_all_insertdelete_hunks(self) -> bool:
        return all(
            hunk.is_insert_hunk or hunk.is_delete_hunk for hunk in self.all_hunks
        )

    def get_diff_by_file_name(self, file_name: str) -> FileDiff:
        for file_diff in self.file_diffs:
            if file_diff.path == file_name:
                return file_diff
        raise ValueError(f"File {file_name} not found in commit")

    def get_hunk_entity_set(
        self, entity_property_name: str, allow_test_file: bool, ignore_statements: bool
    ) -> set[Entity]:
        return {
            entity  # type: ignore
            for file_diff in self.file_diffs
            for hunk in file_diff.hunks
            for entity in getattr(hunk, entity_property_name)
            if allow_test_file or not file_diff.is_test_file
            if not ignore_statements or entity.type != EntityType.STATEMENT  # type: ignore
        }

    def edited_entities(
        self, allow_test_file=True, ignore_statements=True
    ) -> set[Entity]:
        return self.get_hunk_entity_set(
            "edited_entities", allow_test_file, ignore_statements
        )

    def added_entities(
        self, allow_test_file=True, ignore_statements=True
    ) -> set[Entity]:
        return self.get_hunk_entity_set(
            "added_entities", allow_test_file, ignore_statements
        )

    def deleted_entities(
        self, allow_test_file=True, ignore_statements=True
    ) -> set[Entity]:
        return self.get_hunk_entity_set(
            "deleted_entities", allow_test_file, ignore_statements
        )

    def modified_entities(
        self, allow_test_file=True, ignore_statements=True
    ) -> set[Entity]:
        return self.get_hunk_entity_set(
            "modified_entities", allow_test_file, ignore_statements
        )

    def num_edited_entities(self, allow_test_file=True, ignore_statements=True) -> int:
        return len(self.edited_entities(allow_test_file, ignore_statements))

    def num_added_entities(self, allow_test_file=True, ignore_statements=True) -> int:
        return len(self.added_entities(allow_test_file, ignore_statements))

    def num_deleted_entities(self, allow_test_file=True, ignore_statements=True) -> int:
        return len(self.deleted_entities(allow_test_file, ignore_statements))

    def num_modified_entities(
        self, allow_test_file=True, ignore_statements=True
    ) -> int:
        return len(self.modified_entities(allow_test_file, ignore_statements))

    def num_method_entities(self, allow_test_file=True) -> int:
        return sum(
            entity.type == EntityType.METHOD
            for entity in self.edited_entities(allow_test_file)
        )

    def num_function_entities(self, allow_test_file=True) -> int:
        return sum(
            entity.type == EntityType.FUNCTION
            for entity in self.edited_entities(allow_test_file)
        )

    def num_class_entities(self, allow_test_file=True) -> int:
        return sum(
            entity.type == EntityType.CLASS
            for entity in self.edited_entities(allow_test_file)
        )

    def num_statement_entities(self, allow_test_file=True) -> int:
        return sum(
            entity.type == EntityType.STATEMENT
            for entity in self.edited_entities(allow_test_file, ignore_statements=False)
        )

    @property
    def new_files(self) -> list[str]:
        return [file_diff.path for file_diff in self.file_diffs if file_diff.is_new]
