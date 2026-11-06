#!/usr/bin/env python3
"""Run every generated standalone .py headless and report failures."""
from __future__ import annotations

import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {".venv", "scripts", "docs", ".git"}
# Legacy main.py is itself a generator that rewrites the tree on run;
# never execute it in the smoke test.
SKIP_FILES = {"main.py"}


def iter_py_files(root: pathlib.Path):
    for p in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.relative_to(root).name in SKIP_FILES:
            continue
        yield p


def main() -> int:
    files = sorted(iter_py_files(REPO_ROOT))
    fails = 0
    for p in files:
        try:
            result = subprocess.run(
                [sys.executable, str(p)], capture_output=True, text=True, timeout=120
            )
        except subprocess.TimeoutExpired:
            fails += 1
            print(f"FAIL {p.relative_to(REPO_ROOT)}\nTimed out after 120 seconds")
            continue
        except OSError as exc:
            fails += 1
            print(f"FAIL {p.relative_to(REPO_ROOT)}\nCould not start: {exc}")
            continue
        if result.returncode != 0:
            fails += 1
            print(f"FAIL {p.relative_to(REPO_ROOT)}\n{result.stderr[-1500:]}")
    print(f"{len(files) - fails}/{len(files)} scripts passed.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())