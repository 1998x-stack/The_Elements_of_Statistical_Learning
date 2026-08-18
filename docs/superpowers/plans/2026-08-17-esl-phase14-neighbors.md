# ESL Prototype & Nearest Neighbors (Phase 14) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapter 15 (Prototype Methods & Nearest Neighbors, 7 sections): runnable `.py` for every section and a rich `.ipynb` notebook for the one deep section (15.3 k-Nearest-Neighbor Classifiers). Update `progress.yaml`/`README.md`.

**Architecture:** Established pattern — direct-write Chinese `.md` + runnable `.py`; build the deep `.ipynb` via `nbformat`; flip `progress.yaml`; rebuild README with `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib, nbformat.

## Global Constraints

- `.md` narrative in **Chinese**; code/terms English; math in LaTeX. `.md` begins with `# <Full English Title>（<中文名>）`.
- `.py` runs standalone headless (`matplotlib.use("Agg")`), seeded, `if __name__ == "__main__":`, figures saved. **Never run `main.py`.**
- `progress.yaml` keys canonical: `15_Prototype_Methods_and_Nearest_Neighbors/<section>`.
- Statuses `done`/`notebook`; deep this phase: **15.3 only**.
- Suppress benign `RuntimeWarning`s.
- Font preamble for plotting scripts.

## Standard 5-step envelope

- [ ] 1. `.py` 2. `.md` 3. progress 4. run 5. commit

---

### Task 1: 15.1 — Introduction (→ `done`)
`.py` roadmap. `.md`: 基于原型的分类与最近邻思想。`15.1...: stub`→`done`; commit.

### Task 2: 15.2 — Prototype Methods (→ `done`)
`.py`: k 中心点 / 1 中心原型分类（每类一个均值），对比决策。`.md`: 每个类用一个"原型"代表、voronoi 边界。`done`; commit.

### Task 3: 15.3 — k-Nearest-Neighbor Classifiers, **DEEP → notebook** (→ `notebook`)
`.py`: `KNeighborsClassifier` 在二维数据上边界随 k 变化，报告准确率。`.md`: kNN 非参数、k 影响复杂度、与 (4/13) 的联系。 build nb `build_nb_153.py`. `15.3...: stub`→`notebook`; commit.

### Task 4: 15.4 — Adaptive NN Methods (→ `done`)
`.py`: 对 k 做自适应（变面积/判别/局部楯椭）示意。`.md`: 自适应邻域、判别自适应 kNN。`done`; commit.

### Task 5: 15.5 — Computational Considerations (→ `done`)
`.py`: KD-tree 型近邻查询耗时的对比 (naive vs `NearestNeighbors`)。`.md`: 距离查询、KD树/FLANN。`done`; commit.

### Task 6: 15.6 — Bibliographic Notes (→ `done`)
`.py` prints refs. `.md` brief; `done`; commit.

### Task 7: 15.7 — Exercises (→ `done`)
`.py`: k 与距离函数的效应可练习；`.md` 3 个中文习题+参考解。`done`.

### Task 8: README refresh + full verification
- [ ] `python scripts/generate_docs.py --regenerate`
- [ ] `python -m pytest` (≥8) + `python scripts/smoke_test.py`
- [ ] confirm 15.3 → 🔬
- [ ] commit `docs: Phase 14 prototype & neighbors content`

## Self-Review
**Spec coverage:** all 7 ch15 sections authored; deep 15.3 notebook; progress/README. ✓
**Placeholder scan:** every task has concrete `.py` + Chinese `.md`; nb `build_nb_153.py`. ✓
**Type consistency:** 15.3 → notebook; others → done. ✓
**Engineer note:** never run `main.py`; careful with absolute paths; keep kNN cheap for smoke.