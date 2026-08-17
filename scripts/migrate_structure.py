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

# Section-number prefixes that denote a "deep" section (see plan Global Constraints).
DEEP_PREFIXES = {
    "4.5", "4.9", "5.2", "5.4", "5.5", "6.3", "6.4", "7.2", "7.4",
    "9.10", "9.11", "10.5", "10.7", "11.2", "11.4", "12.1", "12.4",
    "12.10", "13.3", "14.2", "14.3", "15.3", "16.3", "16.5", "17.2",
}


def is_deep(section_key: str) -> bool:
    number = section_key.split("_", 1)[0]
    return number in DEEP_PREFIXES


def to_config(node: dict) -> dict:
    """Recursively convert {chapter: {section: []}} into {chapter: {section: meta}}.

    Branch nodes become nested dicts; leaf sections become {"deep": bool}.
    """
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