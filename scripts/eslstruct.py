"""Shared structure-loading/iteration helpers for the generator."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import yaml


def normalize(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def load_structure(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_deep(meta: dict) -> bool:
    return bool(meta.get("deep", False))


def _is_leaf(value) -> bool:
    """A leaf section stores meta whose keys are a subset of {"deep"}.

    A branch stores a dict of child sections (keys contain section names).
    """
    return not isinstance(value, dict) or set(value.keys()).issubset({"deep"})


def iter_sections(structure: dict) -> Iterator[tuple[list[str], str, dict]]:
    """Yield (chapter_parts, section_key, meta) for every leaf section.

    chapter_parts includes the top-level chapter key (and any intermediate
    nesting keys) leading to the leaf.
    """
    def walk(node: dict, parts: list[str]):
        for key, value in node.items():
            if isinstance(value, dict) and not _is_leaf(value):
                yield from walk(value, parts + [key])
            else:
                yield parts, key, value

    yield from walk(structure, [])