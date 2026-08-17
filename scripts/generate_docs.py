#!/usr/bin/env python3
"""Idempotent directory/file generator.

--init        create every scaffold file (.md/.py, .ipynb for deep sections) + README
--regenerate  create any missing scaffold files and rebuild README (default)

Never overwrites an existing authored file.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from eslstruct import load_structure, iter_sections
from eslstruct import is_deep, normalize, progress_key

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE_DIR = pathlib.Path(__file__).resolve().parent / "templates"


def sanitize_section(section: str) -> str:
    return normalize(section)


def _template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def _is_leaf(value) -> bool:
    return not isinstance(value, dict) or set(value.keys()).issubset({"deep"})


def _find_chain(structure: dict, target: str) -> list[str]:
    """Keys from chapter root down to (exclusive of) the section."""

    def walk(node: dict, parts: list[str]):
        for key, value in node.items():
            if isinstance(value, dict) and not _is_leaf(value):
                result = walk(value, parts + [key])
                if result is not None:
                    return result
            else:
                if key == target:
                    return parts
        return None

    result = walk(structure, [])
    return result or []


def section_paths(structure, section: str, _meta: dict, root: pathlib.Path) -> dict[str, pathlib.Path]:
    # On-disk layout is chapter/<section>/<section>.md|.py|.ipynb — the
    # section key is both a directory and the file stem.
    chain = _find_chain(structure, section)  # chapter chain, excluding the section
    sec = sanitize_section(section)
    base = root
    for part in chain:
        base = base / normalize(part)
    base = base / sec
    return {
        "dir": base,
        "md": base / f"{sec}.md",
        "py": base / f"{sec}.py",
        "ipynb": base / f"{sec}.ipynb",
    }


def write_text_if_missing(path: pathlib.Path, content: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def write_md_if_missing(path: str, section: str, chapter: str) -> bool:
    content = _template("md_stub.md.txt").format(chapter=chapter, section=section)
    return write_text_if_missing(pathlib.Path(path), content)


def write_py_if_missing(path: str, section: str) -> bool:
    content = _template("py_stub.py.txt").format(title=section, chapter="", section=section)
    return write_text_if_missing(pathlib.Path(path), content)


def write_ipynb_if_missing(path: str, section: str) -> bool:
    if pathlib.Path(path).exists():
        return False
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {section}\n", "> ⏳ 待补充 — 本节深讲笔记本将在后续阶段撰写。\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [f"# TODO: {section}\n", "print('TODO')\n"],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    import json

    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def status_badge(status: str) -> str:
    return {"stub": "⏳ TODO", "done": "✅ 摘要", "notebook": "🔬 笔记"}.get(status, "⏳ TODO")


def build_readme(structure: dict, progress: dict) -> str:
    lines = [
        "# The Elements of Statistical Learning",
        "",
        "> 一个关于 *The Elements of Statistical Learning* 的目录结构。",
        "",
        "## 目录与进度",
        "",
        "状态图例: ⏳ TODO（未写） · ✅ 摘要（已完成） · 🔬 笔记（含 Jupyter 笔记本）",
        "",
    ]

    def rel_key(parts: list[str], sec: str) -> str:
        """progress.yaml key (chapter/section), no leading slash."""
        return progress_key(parts, sec)

    def file_base(parts: list[str], sec: str) -> str:
        """true on-disk base: chapter/section/section (section is its own dir)."""
        root = "/".join([normalize(p) for p in parts] + [normalize(sec)])
        return root + "/" + normalize(sec)

    def walk(node: dict, parts: list[str]):
        for key, value in node.items():
            if isinstance(value, dict) and not _is_leaf(value):
                lines.append("")
                lines.append(
                    "### " + " / ".join([normalize(p) for p in parts] + [normalize(key)])
                )
                walk(value, parts + [key])
            else:
                rel = rel_key(parts, key)
                status = progress.get(rel, "stub")
                badge = status_badge(status)
                fb = file_base(parts, key)
                lines.append(f"- {badge} **{key}**")
                lines.append(f"  - 摘要: `{fb}.md`")
                lines.append(f"  - 代码: `{fb}.py`")
                if status == "notebook":
                    lines.append(f"  - 笔记: `{fb}.ipynb`")

    walk(structure, [])
    return "\n".join(lines)


def generate_all(structure: dict, root: pathlib.Path = REPO_ROOT) -> dict:
    created = {"md": 0, "py": 0, "ipynb": 0, "skipped": 0}
    for parts, section, meta in iter_sections(structure):
        chapter = "/".join(normalize(p) for p in parts)
        paths = section_paths(structure, section, meta, root)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        if write_md_if_missing(str(paths["md"]), section, chapter):
            created["md"] += 1
        else:
            created["skipped"] += 1
        if write_py_if_missing(str(paths["py"]), section):
            created["py"] += 1
        if is_deep(meta) and not paths["ipynb"].exists():
            write_ipynb_if_missing(str(paths["ipynb"]), section)
            created["ipynb"] += 1
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ESL scaffold generator")
    parser.add_argument("--init", action="store_true", help="create scaffold for all sections")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="alias: create missing files + rebuild README (default)",
    )
    args = parser.parse_args(argv)

    struct = load_structure(REPO_ROOT / "config" / "structure.yaml")
    progress = load_structure(REPO_ROOT / "config" / "progress.yaml")
    created = generate_all(struct, REPO_ROOT)
    print(
        f"created={{md: {created['md']}, py: {created['py']}, "
        f"ipynb: {created['ipynb']}, skipped: {created['skipped']}}}"
    )
    readme = build_readme(struct, progress)
    (REPO_ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(f"README rebuilt ({len(readme)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())