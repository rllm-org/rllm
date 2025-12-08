from pathlib import Path

from r2egym.commit_models.entity_utils import Entity, EntityType
from r2e.models import Class, Function, Module, Identifier, File, Repo
from r2e.pat.dependency_slicer import DependencySlicer, DependencySliceUnparseEnum


def construct_repo_from_path(repo_root: Path):
    repo = Repo(
        repo_org="temp_repo",
        repo_name="temp_repo",
        repo_id=repo_root.as_posix(),
        local_repo_path=repo_root.as_posix(),
    )
    return repo


def construct_file_from_path(file_path: Path, repo: Repo):
    file_module = Module.from_file_path(str(file_path), repo=repo)
    return File(file_module=file_module)


def construct_class_from_entity(entity: Entity, file: File):
    class_name = entity.name
    class_id = Identifier(identifier=f"{file.file_id}.{class_name}")
    return Class(class_id=class_id, class_name=class_name, file=file)


def construct_function_from_entity(entity: Entity, file: File):
    function_name = entity.name
    function_id = Identifier(identifier=f"{file.file_id}.{function_name}")
    return Function(
        function_id=function_id,
        function_name=function_name,
        file=file,
        function_code=entity.content,
    )


def get_funclass_from_entity_reporoot(entity: Entity, repo_root: Path):
    relative_file_path = entity.file_name
    full_file_path = repo_root / relative_file_path

    repo = construct_repo_from_path(repo_root)
    file = construct_file_from_path(full_file_path, repo)

    if entity.type == EntityType.FUNCTION:
        return construct_function_from_entity(entity, file)
    elif entity.type == EntityType.CLASS:
        return construct_class_from_entity(entity, file)


def get_slice_from_funclass(
    funclass_models: list[Class | Function], depth=-1, slice_imports=False
) -> str | None:
    try:
        slicer = DependencySlicer.from_funclass_models(
            funclass_models, depth=depth, slice_imports=slice_imports
        )
        slicer.run()
        dependency_slice = slicer.dependency_graph.unparse(
            DependencySliceUnparseEnum.PATH_COMMENT
        )
        return dependency_slice
    except Exception as e:
        return
