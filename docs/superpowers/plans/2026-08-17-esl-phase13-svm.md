# ESL SVM & Flexible Discriminants (Phase 13) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapter 14 (Support Vector Machines & Flexible Discriminants, 9 sections): runnable `.py` for every section and rich `.ipynb` notebooks for the two deep sections (14.2 Support Vector Classifier, 14.3 SVM & Kernels). Update `progress.yaml`/`README.md`.

**Architecture:** Established pattern — direct-write Chinese `.md` + runnable `.py`; build deep `.ipynb` via `nbformat`; flip `progress.yaml`; rebuild README with `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib, nbformat.

## Global Constraints

- `.md` narrative in **Chinese**; code/terms English; math in LaTeX. `.md` begins with `# <Full English Title>（<中文名>）`.
- `.py` runs standalone headless (`matplotlib.use("Agg")`), seeded, `if __name__ == "__main__":`, figures saved. **Never run `main.py`.**
- `progress.yaml` keys canonical: `14_Support_Vector_Machines_and_Flexible_Discriminants/<section>`.
- Statuses `done`/`notebook`; deep this phase: **14.2, 14.3**.
- Suppress benign `RuntimeWarning`s.
- Font preamble for plotting scripts.

## Standard 5-step envelope

- [ ] 1. `.py` 2. `.md` 3. progress 4. run 5. commit

---

### Task 1: 14.1 — Introduction (→ `done`)
`.py` roadmap. `.md`: 支持向量机作为最大间隔分隔超平面 + 核 `SVM` 两章一体。`14.1...: stub`→`done`; commit.

### Task 2: 14.2 — The Support Vector Classifier, **DEEP → notebook** (→ `notebook`)
`.py`: 用 `svm.LinearSVC` 在 2D 找最大间隔超平面，画支持向量/边界（C 影响软间隔）。`.md`: 最大间隔、soft-margin 罚、支持向量。 build nb `build_nb_142.py`. `14.2...: stub`→`notebook`; commit.

### Task 3: 14.3 — Support Vector Machines & Kernels, **DEEP → notebook** (→ `notebook`)
`.py`: 用 `svm.SVC(kernel=rbf/poly)` 在非线双月数据上展示核边界。`.md`: 核选择、SVM = 特征空间线性分隔。 build nb `build_nb_143.py`. `14.3→notebook`; commit.

### Task 4: 14.4 — Generalizing LDA (→ `done`)
`.py`: LDA 的降秩/判别空间用 `LinearDiscriminantAnalysis(solver='eigen')`。`.md`: 从 LDA 到广义判别。`done`; commit.

### Task 5: 14.5 — Flexible Discriminant Analysis (→ `done`)
`.py`: 用 FDA 思想：把特征先经非线性基（样条）映射再 LDA，展示灵活边界。`.md`: FDA、最优评分、非线性映射+判别。`done`; commit.

### Task 6: 14.6 — Penalized Discriminant Analysis (→ `done`)
`.py`: 正则化判别（岭 LDA）在小样本高维的演示。`.md`: 惩罚 LDA 用于高维。`done`; commit.

### Task 7: 14.7 — Mixture Discriminant Analysis (→ `done`)
`.py`: 高斯混合逐类 + {compute sub-classes}（`MixtureDiscriminantAnalysis` 近似：`GMM` per class）。`.md`: 每类子类混合、MDA。`done`; commit.

### Task 8: 14.8 — Bibliographic Notes (→ `done`)
`.py` prints refs (Vapnik; Boser & Guyon; Cortes; Hastie FDA/MDA). `.md` brief. `done`; commit.

### Task 9: 14.9 — Exercises (→ `done`)
`.py`: (C, kernel, gamma) effect exercisable on make_moons. `.md` 3 Chinese exercises + solved note. `done`.

### Task 10: README refresh + full verification
- [ ] `python scripts/generate_docs.py --regenerate`
- [ ] `python -m pytest` (≥8) + `python scripts/smoke_test.py`
- [ ] confirm 14.2/14.3 → 🔬
- [ ] commit `docs: Phase 13 SVM & flexible discriminants content`

## Self-Review
**Spec coverage:** all 9 ch14 sections authored; deep 14.2/14.3 notebooks; progress/README. ✓
**Placeholder scan:** every task has concrete `.py` + Chinese `.md`; nb builders `build_nb_{142,143}.py`. ✓
**Type consistency:** 14.2/14.3 → notebook; others → done. ✓
**Engineer note:** keep API usage to sklearn built-ins (`LinearSVC`, `SVC`, `LinearDiscriminantAnalysis`); careful with absolute paths; never run `main.py`.