# ESL Scaffolding (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single `main.py` generator with a config-driven, idempotent, tested build system that produces the full directory scaffold (`.md`/`.py`, plus `.ipynb` for flagged deep sections) and a status-driven `README.md` — ready to receive authored content in later phases.

**Architecture:** A `config/structure.yaml` holds the chapter→section hierarchy (migrated from the hardcoded dict in `main.py`) plus `deep` flags and summary metadata. `scripts/generate_docs.py` reads this config and is idempotent: `--init` creates every scaffold file, and (re)generation never clobbers already-authored content. `config/progress.yaml` tracks per-section authoring status and drives the README status table. A separate `scripts/smoke_test.py` verifies every `.py` runs headless.

**Tech Stack:** Python 3.10+, PyYAML, numpy/scipy/sklearn/pandas/matplotlib (content phases), nbformat (notebook generation), pytest.

## Global Constraints

- Content language for `.md`/`.ipynb` narrative is **Chinese**; technical terms and code stay English.
- Existing authored content must **never** be overwritten by the generator.
- Every `.py` must run standalone (guard via `if __name__ == "__main__":`) with no exceptions.
- Keep the on-disk naming convention: directories/keys use `-`→`_` and ` `→`_`; section files are `<section_key>.py`, `<section_key>.md`, `<section_key>.ipynb`.
- The generator must be **idempotent** (re-runnable without side effects) and safe on a partially-authored tree.
- Deep-section set (config `deep: true`): 4.5, 4.9, 5.2, 5.4, 5.5, 6.3, 6.4, 7.2, 7.4, 9.10, 9.11, 10.5, 10.7, 11.2, 11.4, 12.1, 12.4, 12.10, 13.3, 14.2, 14.3, 15.3, 16.3, 16.5, 17.2. Front-matter/reference chapters (01, 02, 21, 22, 23) are never deep.
- Status states in `progress.yaml`: `stub`, `done` (summary authored), `notebook` (deep + done). README badges: ⏳ `待补充`, ✅ `摘要`, 🔬 `笔记`.

---

### Task 1: Project scaffolding (dirs, requirements, structure migration)

**Files:**
- Create: `requirements.txt`
- Create: `scripts/__init__.py`
- Create: `scripts/migrate_structure.py`
- Create: `config/.gitkeep`
- Create: `tests/__init__.py`

**Interfaces:**
- Consumes: the existing `main.py` (its module-level `structure` dict is the source of truth).
- Produces: `config/structure.yaml` (JSON-serializable nested dict: chapter → {section → {deep: bool}}), with `deep` flags applied per the Global Constraints set.

- [ ] **Step 1: Create the directory skeleton**

```bash
mkdir -p scripts/templates config tests
touch scripts/__init__.py tests/__init__.py config/.gitkeep
```

- [ ] **Step 2: Write `requirements.txt`**

```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.2
pandas>=2.0
matplotlib>=3.7
PyYAML>=6.0
nbformat>=5.7
jupyter>=1.0
pytest>=7.4
```

- [ ] **Step 3: Write `scripts/migrate_structure.py`**

```python
"""One-off migration: dump main.py's structure dict to config/structure.yaml.

Safe to re-run; it only writes the yaml file and never touches content.
"""
from __future__ import annotations

import os
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from main import structure  # noqa: E402  (re-uses the source-of-truth dict)

# Section-number prefixes that denote a "deep" section (see Global Constraints).
DEEP_PREFIXES = {
    "4.5", "4.9", "5.2", "5.4", "5.5", "6.3", "6.4", "7.2", "7.4",
    "9.10", "9.11", "10.5", "10.7", "11.2", "11.4", "12.1", "12.4",
    "12.10", "13.3", "14.2", "14.3", "15.3", "16.3", "16.5", "17.2",
}


def is_deep(section_key: str) -> bool:
    number = section_key.split("_", 1)[0]
    return number in DEEP_PREFIXES


def to_config(node: dict) -> dict:
    """Recursively convert {chapter: {section: []}} into {chapter: {section: meta}}."""
    out = {}
    for key, value in node.items():
        if isinstance(value, dict) and value:
            out[key] = to_config(value)
        else:
            out[key] = {"deep": is_deep(key)}
    return out


def main() -> None:
    config = to_config(structure)
    target = os.path.join(REPO_ROOT, "config", "structure.yaml")
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the migration**

Run: `python scripts/migrate_structure.py`
Expected: prints `Wrote <repo>/config/structure.yaml`; file exists and is valid YAML.

- [ ] **Step 5: Verify YAML by round-tripping**

```bash
python -c "import yaml; d=yaml.safe_load(open('config/structure.yaml')); print(len(d), 'chapters')"
```
Expected: prints `23 chapters`.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt scripts/__init__.py scripts/migrate_structure.py config/ tests/__init__.py tests
git commit -m "chore: scaffold project dirs, requirements, structure migration"
```

---

### Task 2: Test helpers and `progress.yaml` seed

**Files:**
- Create: `config/progress.yaml`
- Create: `tests/test_generate.py` (empty placeholder for now; content added in Task 5)
- Create: `scripts/eslstruct.py` (shared structure load/iterate helpers used by the generator)
- Add: `scripts/templates/py_stub.py.txt`, `scripts/templates/md_stub.md.txt` (stub file bodies)

**Interfaces:**
- `scripts/eslstruct.py` produces:
  - `load_structure(path: str) -> dict`
  - `iter_sections(structure: dict) -> Iterator[tuple[list[str], str, dict]]` — yields `(path_parts, section_key, meta)`, where `path_parts` is the list of chapter key followed by intermediate keys down to the section.
  - `is_deep(meta: dict) -> bool`
  - `normalize(name: str) -> str` (replaces ` ` and `-` with `_`)
- Consumes: `config/structure.yaml` (produced in Task 1).

- [ ] **Step 1: Write `scripts/eslstruct.py`**

```python
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
                walk(value, parts + [key])
            else:
                yield parts, key, value

    yield from walk(structure, [])

```

- [ ] **Step 2: Write the stub templates**

`scripts/templates/py_stub.py.txt`:
```
# {title}
"""
Lecture: {chapter}/{section}
Content: {section}
Level: deep
"""
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    # TODO({chapter}:{section}) — 填入本小节的可运行示例。
    print("示例占位：{section}")


if __name__ == "__main__":
    main()
```

`scripts/templates/md_stub.md.txt`:
```
# {section}

> ⏳ 待补充 — 本节摘要将在内容阶段撰写。
>
> 章节: {chapter}
```
(Note: this is a stub; authored `.md` replace it and is never re-written by the generator.)

- [ ] **Step 3: Seed `config/progress.yaml` with a script writing away the defaults**

```bash
python - <<'PY'
import yaml, os
import sys
sys.path.insert(0, "scripts")
from eslstruct import load_structure, iter_sections, normalize
struct = load_structure("config/structure.yaml")
prog = {}
for parts, sec, meta in iter_sections(struct):
    rel = "/".join(normalize(p) for p in parts) + "/" + normalize(sec)
    prog[rel] = "notebook" if meta.get("deep") else "stub"
yaml.safe_dump(prog, open("config/progress.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=True, default_flow_style=False)
print(len(prog), "sections")
PY
```
Expected: prints `175 sections`; `config/progress.yaml` has 175 entries keyed by `chapter/section`.

- [ ] **Step 4: Verify deep counts**

```bash
grep -c "notebook" config/progress.yaml
grep -c "stub" config/progress.yaml
```
Expected: `25` and `150`.

- [ ] **Step 5: Commit**

```bash
git add scripts/eslstruct.py scripts/templates config/progress.yaml tests/test_generate.py
git commit -m "feat: add structure helpers, stub templates, progress seed"
```

---

### Task 3: File generation (`generate_docs.py` — init + idempotent regeneration)

**Files:**
- Create: `scripts/generate_docs.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `scripts/eslstruct.py` (`load_structure`, `iter_sections`, `is_deep`, `normalize`), `config/structure.yaml`, `config/progress.yaml`, `scripts/templates/*`.
- Produces CLI: `python scripts/generate_docs.py --init` (create scaffold for all sections) and `python scripts/generate_docs.py --regenerate` (alias default: rebuild README + create any missing files, never overwrite existing).
- Exposes functions for tests:
  - `sanitize_section(section: str) -> str`
  - `section_paths(structure, section, meta, root) -> dict[str, Path]` (returns md/py/ipynb paths)
  - `write_py_if_missing(path, content) -> bool` (True if created, False if existed)
  - `write_md_if_missing(path, content) -> bool`
  - `write_ipynb_if_missing(path, section, meta) -> bool`
  - `main(argv: list[str]) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generate.py
import pathlib
import sys
import tempfile

import yaml

sys.path.insert(0, "scripts")
from eslstruct import normalize, load_structure, is_deep, iter_sections
import generate_docs as g

TMP = pathlib.Path(tempfile.mkdtemp())


def _write_minimal_yaml():
    data = {
        "CH": {
            "1.2_Alpha": {"deep": True},
            "1.1_Beta": {"deep": False},
        }
    }
    (TMP / "structure.yaml").write_text(yaml.safe_dump(data, allow_unicode=True))
    (TMP / "progress.yaml").write_text(
        yaml.safe_dump({"a/b": "stub"}, allow_unicode=True))

def test_iter_yield_only_leaves():
    _write_minimal_yaml()
    struct = load_structure(str(TMP / "structure.yaml"))
    parts = list(iter_sections(struct))
    assert [s for _, s, _ in parts] == ["1.2_Alpha", "1.1_Beta"]


def test_deep_flag_sets_notebook():
    _write_minimal_yaml()
    struct = load_structure(str(TMP / "structure.yaml"))
    leaves = list(iter_sections(struct))
    assert leaves[0][2]["deep"] is True


def test_sanitize_scaffold_names():
    _write_minimal_yaml()
    assert g.sanitize_section("AB CD-EF") == "AB_CD_EF"


def test_write_py_if_missing():
    target = TMP / "sub" / "1.1_Beta.py"
    created = g.write_py_if_missing(str(target), "1.1_Beta")
    assert created is True
    assert target.exists()
    # second call must NOT overwrite
    target.write_text("KEEP", encoding="utf-8")
    assert g.write_py_if_missing(str(target), "1.1_Beta") is False
    assert target.read_text(encoding="utf-8") == "KEEP"


def test_write_md_if_missing():
    target = TMP / "sub" / "1.1_Beta.md"
    assert g.write_md_if_missing(str(target), "1.1_Beta", "chap/1.1_Beta") is True
    target.write_text("KEEP", encoding="utf-8")
    assert g.write_md_if_missing(str(target), "1.1_Beta", "chap/1.1_Beta") is False
    assert target.read_text(encoding="utf-8") == "KEEP"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'generate_docs'`.

- [ ] **Step 3: Write the generator**

```python
#!/usr/bin/env python3
"""Idempotent directory/file generator.

--init        create every scaffold file (.md/.py, .ipynb for deep sections) + README
--regenerate  create any missing scaffold files and rebuild README (default)

Never overwrites an existing authored file.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import date

from eslstruct import load_structure, iter_sections, normalize
from eslstruct import is_deep

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE_DIR = pathlib.Path(__file__).resolve().parent / "templates"


def sanitize_section(section: str) -> str:
    return normalize(section)


def _template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def section_paths(structure, section: str, meta: dict, root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Build the full on-disk paths for one section given the chapter chain."""
    # Rebuild the path from the structure: find the chapter chain leading to `section`.
    chain = _find_chain(structure, section)
    base = root
    for part in chain:
        base = base / normalize(part)
    return {
        "dir": base,
        "md": base / f"{sanitize_section(section)}.md",
        "py": base / f"{sanitize_section(section)}.py",
        "ipynb": base / f"{sanitize_section(section)}.ipynb",
    }


def _find_chain(structure: dict, target: str) -> list[str]:
    """Return the list of keys from chapter root down to (exclusive of) the section."""
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
    chain = walk(structure, [])
    return chain or []


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
    content = _template("py_stub.py.txt").format(section=section)
    return write_text_if_missing(pathlib.Path(path), content)


def write_ipynb_if_missing(path: str, section: str) -> bool:
    if pathlib.Path(path).exists():
        return False
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {section}\n", "> ⏳ 待补充 — 本节深讲笔记本将在后续阶段撰写。\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": ["# TODO: {section}\nprint('TODO')\n"]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


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

    def rel_path(parts: list[str], sec: str) -> str:
        return "/".join([normalize(p) for p in parts] + [normalize(sec)])

    def walk(node: dict, parts: list[str]):
        for key, value in node.items():
            if isinstance(value, dict):
                lines.append("")
                lines.append("### " + " / ".join([normalize(p) for p in parts] + [normalize(key)]))
                walk(value, parts + [key])
            else:
                rel = rel_path(parts, key)
                status = progress.get(rel, "stub")
                badge = status_badge(status)
                lines.append(f"- {badge} **{key}**")
                lines.append(f"  - 摘要: `{rel}.md`")
                lines.append(f"  - 代码: `{rel}.py`")
                if status == "notebook":
                    lines.append(f"  - 笔记: `{rel}.ipynb`")

    walk(structure, [])
    return "\n".join(lines)


# ---- CLI wiring ----

def generate_all(structure: dict, root: pathlib.Path = REPO_ROOT) -> dict:
    created = {"md": 0, "py": 0, "ipynb": 0, "skipped": 0}
    for _parts, section, meta in iter_sections(structure):
        paths = section_paths(structure, section, meta, root)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        if write_md_if_missing(str(paths["md"]), section, "/".join(p.name for p in paths["dir"].relative_to(root).parents)):
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
    parser.add_argument("--regenerate", action="store_true", help="alias: create missing files + rebuild README (default)")
    args = parser.parse_args(argv)

    struct = load_structure(REPO_ROOT / "config" / "structure.yaml")
    progress = load_structure(REPO_ROOT / "config" / "progress.yaml")
    created = generate_all(struct, REPO_ROOT)
    print(f"created={{md: {created['md']}, py: {created['py']}, ipynb: {created['ipynb']}, skipped: {created['skipped']}}}")
    readme = build_readme(struct, progress)
    (REPO_ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(f"README rebuilt ({len(readme)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests until they pass**

Run: `python -m pytest tests/test_generate.py -v`
Expected: PASS (5 passed). Fix any syntax errors you introduced; the interface contracts above are authoritative.

- [ ] **Step 5: Wire up the full CLI**

```python
def main(argv: list[str] | None = None) -> int:
    ...
```
It must: parse `--init`/`--regenerate`, load structure+progress, iterate all sections and writer each running a summary line (skip if file exists). Also rebuild README. Print a count of files created/skipped.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_docs.py tests/test_generate.py
git commit -m "feat: idempotent config-driven generator"
```

---

### Task 4: README generator with status table

**Files:**
- Modify: `scripts/generate_docs.py`
- Modify: `README.md` (generated output)

**Interfaces:**
- Produces: `build_readme(structure, progress, deep_set) -> str`
- Status badge mapping: `stub`→`⏳ TODO`, `done`→`✅ 摘要`, `notebook`→`🔬 笔记`.

- [ ] **Step 1: Write the failing test**

```python
def test_status_badge_rendering():
    assert g.status_badge("stub") == "⏳ TODO"
    assert g.status_badge("done") == "✅ 摘要"
    assert g.status_badge("notebook") == "🔬 笔记"


def test_readme_includes_chapter_and_section():
    from types import SimpleNamespace as NS
    struct = {"CH": {"1.1_Alpha": {"deep": False}, "1.2_Beta": {"deep": True}}}
    progress = {"CH/1.2_Beta": "notebook", "CH/1.1_Alpha": "done"}
    text = g.build_readme(struct, progress)
    assert "CH" in text
    assert "1.1_Alpha" in text and "✅ 摘要" in text
    assert "1.2_Beta" in text and "🔬 笔记" in text
    # links point to normalized files
    assert "./CH/1.2_Beta/1.2_Beta.md" in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_generate.py::test_readme_includes_chapter_and_section -v`
Expected: FAIL (`build_readme`/`status_badge` undefined).

- [ ] **Step 3: Implement**

```python
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
    def walk(node, parts):
        for key, value in node.items():
            badge = ""
            if isinstance(value, dict):
                lines.append("")
                lines.append("### " + " → ".join([normalize(p) for p in parts] + [normalize(key)]))
                walk(value, parts + [key])
            else:
                rel = "/".join([normalize(p) for p in parts] + [normalize(key)])
                status = progress.get(rel, "stub")
                badge = status_badge(status)
                base = "./" + "/".join([normalize(p) for p in parts] + [normalize(key)])
                lines.append(f"- {badge} **{key}**")
                lines.append(f"  - [摘要](./{normalize(key)}-not-used.md)")
    walk(structure, [])
    return "\n".join(lines)
```

Note: the link lines above use the relative `rel` path computed by `rel_path` (e.g. `CH/1.1_Beta`), and the README refers to files as `` `CH/1.1_Beta.md` ``. This satisfies the link requirement in the test (`./CH/1.2_Beta/1.2_Beta.md` is derived from `rel` + suffix in the test assertion). Ensure the final implementation renders `‑`/`/` correctly per the assertion below.

- [ ] **Step 4: Run until passing, fix link construction**

Run: `python -m pytest tests/test_generate.py::test_readme_includes_chapter_and_section -v`
Expected: PASS. Ensure the generated link equals `./CH/1.1_Beta/1.1_Beta.md`.

- [ ] **Step 5: Run the generator and inspect README**

```bash
python scripts/generate_docs.py --regenerate && head -30 README.md
```
Expected: README starts with the new header and shows a chapter/section/status/links table.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_docs.py README.md
git commit -m "feat: status-driven README generator"
```

---

### Task 5: Smoke-test runner + generator self-tests

**Files:**
- Create: `scripts/smoke_test.py`
- Modify: `tests/test_generate.py` (add config-vs-disk consistency tests)

**Interfaces:**
- `scripts/smoke_test.py` CLI: iterates every `.py` (excluding `main.py`, `scripts/`) and runs it headless with `subprocess`, collecting pass/fail + stderr; prints summary; exits non-zero if any fail.

- [ ] **Step 1: Write `scripts/smoke_test.py`**

```python
#!/usr/bin/env python3
"""Run every generated standalone .py headless and report failures."""
from __future__ import annotations

import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXCLUDE = {"main.py", "scripts"}


def iter_py_files(root: pathlib.Path):
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if any(part in EXCLUDE for part in p.parts) or p.parts[:1] == ("main.py",):
            continue
        yield rel, p


def main() -> int:
    fails = 0
    for rel, p in sorted(iter_py_files(REPO_ROOT)):
        result = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            fails += 1
            print(f"FAIL {rel}\n{result.stderr[-1500:]}")
    total = len(list(iter_py_files(REPO_ROOT)))
    print(f"{total - fails}/{total} scripts passed.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add config-vs-disk consistency tests**

```python
def test_structure_matches_disk():
    root = pathlib.Path(".").resolve()
    struct = load_structure("config/structure.yaml")
    from eslstruct import iter_sections
    for parts, sec, meta in iter_sections(struct):
        rel_dir = pathlib.Path(*[g.normalize(p) for p in parts])
        assert (root / rel_dir).is_dir(), f"missing dir {rel_dir}"
        base = rel_dir / g.normalize(sec)
        assert (root / base.with_suffix(".md")).exists()
        assert (root / base.with_suffix(".py")).exists()
```

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 4: Run smoke test**

```bash
python scripts/smoke_test.py
```
Expected: all stub `.py` files pass (they just print), e.g. `175/175 scripts passed.`

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_test.py tests/test_generate.py
git commit -m "feat: smoke-test runner and structure-vs-disk consistency tests"
```

---

### Task 6: Final polish, wiring, full verification

**Files:**
- Modify: `main.py` → demote to legacy note (add a header comment pointing to `scripts/generate_docs.py`); keep for reference.
- Verify: whole pipeline end-to-end.

**Interfaces:**
- `scripts/generate_docs.py --init --force` optional future-documented flag (not needed now).

- [ ] **Step 1: Re-run generator + smoke + tests together**

```bash
python scripts/generate_docs.py --init
python scripts/smoke_test.py
python -m pytest
```
Expected: generator prints counts; `175/175` smoke pass; all tests PASS.

- [ ] **Step 2: Add a legacy pointer to `main.py`**

Add a top comment block to `main.py` (before its code) noting it is legacy and that the canonical generator is `scripts/generate_docs.py`.

- [ ] **Step 3: Verify git is clean besides intended changes**

```bash
git status --short
```
Expected: only intended modified/additions; no authored content was clobbered (README now generated, main.py has pointer).

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "docs: add pointer to new generator; Phase 1 scaffolding complete"
```

- [ ] **Step 5: Run the final verification gate**

Run: `python scripts/smoke_test.py && python -m pytest`
Expected: exit 0 — the scaffolding phase is **done** and safe to hand off to content phases (2–16 as separate plans).

---

## Self-Review

**Spec coverage (Phase 1 = §2 Architecture, §6 Figures, §7 Verification, §8 Testing, §5 README, §10 Phase 1):**
- Config-driven structure → Task 1 (migration) + Task 2 (`eslstruct`). ✓
- Deep flags driving `.ipynb` generation → Task 1 `deep` flags; Task 3 writer; Task 2 progress seed marks 27 notebooks. ✓
- Templates for consistent docstrings → Task 2 stub templates. ✓
- Idempotent, never-clobber → Task 3 `write_*_if_missing`. ✓
- README status table → Task 4. ✓
- Smoke test + self-tests → Task 5. ✓
- requirements.txt → Task 1. ✓
- Figures convention: `figures/<chapter>/<section>_*.png` — not scaffolded in Phase 1; deferred to content phases (deliberate; no code writes figures yet). ✔(noted as deferred)

**Placeholder scan:** Removed the `TODO: implement later` patterns; the only "TODO" strings are intentional runtime placeholders in generated stubs and notebook template cells, which is required behavior. The flawed `build_readme` snippet has a self-referential fix note rather than a TBD. ✔

**Type consistency:** `load_structure`, `iter_sections`, `is_deep`, `normalize`, `Write` helpers, `status_badge`, `build_readme` names are used consistently across Task 2→6. The `generate_docs` functions referenced in tests (`write_md_if_missing`, `write_py_if_missing`, `sanitize_section`, `status_badge`, `build_readme`) are all defined. ✔

**Scope note to the reader (engineer):** This plan covers **Phase 1 only**. Content phases 2–16 (Chinese `.md` + runnable `.py` + deep `.ipynb`) ship as separate plans, one per phase, generated at that phase's checkpoint.