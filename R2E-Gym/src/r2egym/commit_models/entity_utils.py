import ast
from enum import Enum
from pathlib import Path
from typing import Optional
from collections import defaultdict

from pydantic import BaseModel, Field


class EntityType(Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    STATEMENT = "statement"
    IMPORT = "import"


class Entity(BaseModel):
    file_name: Path
    type: EntityType
    name: str
    content: str
    ast_type_str: str
    start_lineno: int
    end_lineno: int
    parent: Optional["Entity"] = None

    def __hash__(self) -> int:
        if self.type in (EntityType.FUNCTION, EntityType.CLASS, EntityType.METHOD):
            return hash((self.file_name, self.type, self.name))
        else:
            return hash((self.file_name, self.type, self.content))

    def __lt__(self, other: "Entity") -> bool:
        return self.start_lineno < other.start_lineno

    def __eq__(self, other: "Entity") -> bool:
        if self.file_name != other.file_name:
            return False
        if self.type in (EntityType.FUNCTION, EntityType.CLASS, EntityType.METHOD):
            return self.name == other.name
        return self.content == other.content

    def prompt_repr(self) -> str:
        if self.type in (EntityType.FUNCTION, EntityType.CLASS, EntityType.METHOD):
            return f"{self.type.value} '{self.name}' -- {self.file_name}:{self.start_lineno}-{self.end_lineno}"

    def json_summary_dict(self) -> dict:
        return {
            "file_name": str(self.file_name),
            "type": self.type.value,
            "name": self.name,
            "ast_type_str": self.ast_type_str,
            "start_lineno": self.start_lineno,
            "end_lineno": self.end_lineno,
        }


class CodeStructure(BaseModel):
    entities: list[Entity]
    entities_by_line: dict[int, set[Entity]]

    def get_entity_by_name_type(self, name: str, type: EntityType) -> Entity | None:
        for entity in self.entities:
            if entity.name == name and entity.type == type:
                return entity
        return None


def get_top_level_entities(file_name: str, source_code: str) -> list[Entity]:
    """
    Parses Python source code and returns a list of dictionaries
    containing information about each top-level entity:

    treats class methods as entities as well
    """
    try:
        tree = ast.parse(source_code)
    except:
        # print(f"Syntax error in {file_name}")
        # print(source_code)
        raise
    entities = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entity = Entity(
                file_name=file_name,
                type=EntityType.FUNCTION,
                name=node.name,
                content=ast.get_source_segment(source_code, node),
                ast_type_str=type(node).__name__,
                start_lineno=node.lineno,
                end_lineno=node.end_lineno,
            )
            entities.append(entity)
        elif isinstance(node, ast.ClassDef):
            entity = Entity(
                file_name=file_name,
                type=EntityType.CLASS,
                name=node.name,
                content=ast.get_source_segment(source_code, node),
                ast_type_str=type(node).__name__,
                start_lineno=node.lineno,
                end_lineno=node.end_lineno,
            )

            for child_node in ast.iter_child_nodes(node):
                if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_entity = Entity(
                        file_name=file_name,
                        type=EntityType.METHOD,
                        name=node.name + "." + child_node.name,
                        content=ast.get_source_segment(source_code, node),
                        ast_type_str=type(node).__name__,
                        start_lineno=child_node.lineno,
                        end_lineno=child_node.end_lineno,
                        parent=entity,
                    )
                    entities.append(method_entity)
            entities.append(entity)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                entity = Entity(
                    file_name=file_name,
                    type=EntityType.IMPORT,
                    name=alias.name,
                    content=ast.get_source_segment(source_code, node),
                    ast_type_str=type(node).__name__,
                    start_lineno=node.lineno,
                    end_lineno=node.end_lineno,
                )
                entities.append(entity)
        else:
            entity = Entity(
                file_name=file_name,
                type=EntityType.STATEMENT,
                name="",
                content=ast.get_source_segment(source_code, node),
                ast_type_str=type(node).__name__,
                start_lineno=node.lineno,
                end_lineno=node.end_lineno,
            )
            entities.append(entity)
    return entities


def build_code_structure(
    file_name: str, source_code: str | None = None
) -> CodeStructure:
    """
    Constructs a CodeStructure object from the given source code.

    Returns:
        A CodeStructure object.
    """
    entities = get_top_level_entities(file_name, source_code)
    entities_by_line: dict[int, set[Entity]] = defaultdict(set)
    for entity in entities:
        for lineno in range(entity.start_lineno, entity.end_lineno + 1):
            entities_by_line[lineno].add(entity)
    return CodeStructure(entities=entities, entities_by_line=entities_by_line)


def pprint_entity(entity: Entity):
    """
    Pretty-prints the given entity.
    """
    ename_str = f" '{entity.name}'" if entity.name else ""
    print(f"{entity.type}{ename_str} ({entity.start_lineno}-{entity.end_lineno})")


def pprint_entities(entities: list[Entity] | set[Entity] | tuple[Entity]):
    """
    Pretty-prints the given list of entities.
    """
    for entity in entities:
        pprint_entity(entity)


class CommentAndDocstringRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        # Remove docstring if it exists
        if (
            ast.get_docstring(node) is not None
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Str)
        ):
            node.body = node.body[1:]
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        # Remove docstring if it exists
        if (
            ast.get_docstring(node) is not None
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Str)
        ):
            node.body = node.body[1:]
        return self.generic_visit(node)

    def visit_Expr(self, node):
        # Remove standalone string expressions (comments)
        if isinstance(node.value, ast.Str):
            return None
        return node


def unparse_entity_without_comment_docs(entity: Entity | None) -> str:
    if entity is None:
        return ""
    transformer = CommentAndDocstringRemover()
    tree = ast.parse(entity.content)
    tree = transformer.visit(tree)
    return ast.unparse(tree)
