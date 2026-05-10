"""Compatibility wrapper for the filtered SWE-smith mix preparation script."""

from swe.scripts.prepare_filtered_mix import (
    EVAL_TYPE,
    SOURCE_DATASETS,
    WORKING_DIR,
    get_docker_image,
    instance_to_task,
    main,
    prepare_filtered_mix,
    serialize_complex_fields,
)

__all__ = [
    "EVAL_TYPE",
    "SOURCE_DATASETS",
    "WORKING_DIR",
    "get_docker_image",
    "instance_to_task",
    "main",
    "prepare_filtered_mix",
    "serialize_complex_fields",
]


if __name__ == "__main__":
    main()
