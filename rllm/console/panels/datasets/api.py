"""Datasets panel API.

* ``GET /``                                — every dataset as a card row.
* ``GET /categories``                      — category names for filter pills.
* ``GET /{name}``                          — full detail incl. per-split counts.
* ``GET /{name}/entries?split=&offset=&limit=`` — paginated row reader.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from rllm.console.panels.datasets import loader

router = APIRouter()

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_name(name: str) -> str:
    if not _SAFE_NAME.match(name):
        raise HTTPException(400, "Bad dataset name")
    return name


@router.get("")
@router.get("/")
def list_datasets() -> dict[str, Any]:
    return {
        "datasets": loader.list_datasets(),
        "categories": loader.categories(),
    }


@router.get("/categories")
def list_categories() -> list[str]:
    return loader.categories()


@router.get("/{name}")
def get_dataset(name: str) -> dict[str, Any]:
    _validate_name(name)
    detail = loader.get_dataset(name)
    if detail is None:
        raise HTTPException(404, f"unknown dataset: {name}")
    return detail


@router.get("/{name}/entries")
def get_entries(
    name: str,
    split: str = Query(..., min_length=1),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=200),
) -> dict[str, Any]:
    _validate_name(name)
    if not _SAFE_NAME.match(split):
        raise HTTPException(400, "Bad split name")
    try:
        return loader.get_entries(name, split=split, offset=offset, limit=limit)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
