# tests/test_generate.py
import pathlib
import sys
import tempfile

import yaml

sys.path.insert(0, "scripts")
from eslstruct import load_structure, is_deep, iter_sections, normalize

import generate_docs as g

TMP = pathlib.Path(tempfile.mkdtemp())


def _write_minimal_yaml():
    data = {
        "CH": {
            "1.1_Beta": {"deep": False},
            "1.2_Alpha": {"deep": True},
        }
    }
    (TMP / "structure.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    (TMP / "progress.yaml").write_text(yaml.safe_dump({"a/b": "stub"}, allow_unicode=True), encoding="utf-8")


def test_iter_yield_only_leaves():
    _write_minimal_yaml()
    struct = load_structure(str(TMP / "structure.yaml"))
    parts = list(iter_sections(struct))
    assert [s for _, s, _ in parts] == ["1.1_Beta", "1.2_Alpha"]


def test_deep_flag_sets_notebook():
    _write_minimal_yaml()
    struct = load_structure(str(TMP / "structure.yaml"))
    leaves = list(iter_sections(struct))
    assert leaves[1][2]["deep"] is True


def test_sanitize_scaffold_names():
    assert g.sanitize_section("AB CD-EF") == "AB_CD_EF"


def test_write_py_if_missing():
    target = TMP / "sub"
    target.mkdir(parents=True, exist_ok=True)
    py = target / "1.1_Beta.py"
    created = g.write_py_if_missing(str(py), "1.1_Beta")
    assert created is True
    assert py.exists()
    # second call must NOT overwrite
    py.write_text("KEEP", encoding="utf-8")
    assert g.write_py_if_missing(str(py), "1.1_Beta") is False
    assert py.read_text(encoding="utf-8") == "KEEP"


def test_write_md_if_missing():
    md = TMP / "1.1_Beta.md"
    assert g.write_md_if_missing(str(md), "1.1_Beta", "chap") is True
    md.write_text("KEEP", encoding="utf-8")
    assert g.write_md_if_missing(str(md), "1.1_Beta", "chap") is False
    assert md.read_text(encoding="utf-8") == "KEEP"


def test_status_badge_rendering():
    assert g.status_badge("stub") == "⏳ TODO"
    assert g.status_badge("done") == "✅ 摘要"
    assert g.status_badge("notebook") == "🔬 笔记"


def test_readme_includes_chapter_and_section():
    struct = {"CH": {"1.1_Alpha": {"deep": False}, "1.2_Beta": {"deep": True}}}
    progress = {"CH/1.1_Alpha": "done", "CH/1.2_Beta": "notebook"}
    text = g.build_readme(struct, progress)
    assert "CH" in text
    assert "1.1_Alpha" in text and "✅ 摘要" in text
    assert "1.2_Beta" in text and "🔬 笔记" in text
    assert "`CH/1.2_Beta/1.2_Beta.md`" in text
    assert "`CH/1.2_Beta/1.2_Beta.ipynb`" in text