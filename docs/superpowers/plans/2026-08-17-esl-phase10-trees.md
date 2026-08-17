# ESL Additive Models & Trees (Phase 10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapter 11 (Additive Models, Trees, & Related Methods, 9 sections): runnable `.py` for every section and rich `.ipynb` notebooks for the two deep sections (11.2 Tree-Based Methods, 11.4 MARS). Update `progress.yaml`/`README.md`.

**Architecture:** Established pattern — direct-write Chinese `.md` + runnable `.py`; build deep `.ipynb` via `nbformat`; flip `progress.yaml`; rebuild README with `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib, nbformat.

## Global Constraints

- `.md` narrative in **Chinese**; code/terms English; math in LaTeX. `.md` begins with `# <Full English Title>（<中文名>）`.
- `.py` runs standalone headless (`matplotlib.use("Agg")`), seeded, `if __name__ == "__main__":`, figures saved. **Never run `main.py`.*
- `progress.yaml` keys canonical: `11_Additive_Models,_Trees,_and_Related_Methods/<section>`.
- Statuses `done`/`notebook`; deep this phase: **11.2** (Tree-Based Methods), **11.4** (MARS).
- Note `MARS` section key contains `:` in name: `11.4_MARS:_Multivariate_Adaptive_Regression_Splines`. Preserve.
- Suppress benign `RuntimeWarning`s.
- Font preamble for plotting scripts.

## Standard 5-step envelope

- [ ] 1. `.py` 2. `.md` 3. progress 4. run 5. commit

---

### Task 1: 11.1 — Generalized Additive Models (→ `done`)
`.py`: fit simple GAM via backfitting for 2-input additive data (fit each dimension with smoothing splines / basis). `.md`: GAM、backfitting加性函数可解释性。`11.../11.1...: stub`→`done`; commit.

### Task 2: 11.2 — Tree-Based Methods, **DEEP → `notebook`** (→ `notebook`)
`.py`: fit `DecisionTreeRegressor`/`DecisionTreeClassifier` on synthetic (2D) data, print depth/nodes + plot splits; show tree regression step function. `.md`: 二叉树递归划分 (Gini/MSE)、不稳定性、加性可解释。build nb `build_nb_112.py`. `11.2...: stub`→`notebook`; commit.

### Task 3: 11.3 — PRIM: Bump Hunting (→ `done`)
`.py`: implement fast simple box-search (PRIM-ish) on 2D to find high-output region; `.md`: PRIM 的目标盒、盒平均响应、扰动/剥离。`done`; commit.

### Task 4: 11.4 — MARS, **DEEP → `notebook`** (→ `notebook`)
`.py`: `sklearn` 无 MARS, 用手动前向 hinge 基(截断线性)在 1D 上逐步加基并用 GCV 选折点。`.md`: 乘积基、前向加入 Hinge、选节点。build nb `build_nb_114.py`. `11.4...: stub`→`notebook`.

### Task 5: 11.5 — Hierarchical Mixtures of Experts (→ `done`)
`.py`: small 2-expert gate/softmax mixture demo (`GaussianMixture`-style), plot gate. `.md`: 门控、专家分层、软加权. `done`; commit.

### Task 6: 11.6 — Missing Data (→ `done`)
`.py`: 演示均值填入、missforest/EM 简单 imputation 效果. `.md`: 缺失机制 MCAR/MAR/MNAR、单/多插补. `done`; commit.

### Task 7: 11.7 — Computational Considerations (→ `done`)
`.py`: timing tree/Forest grow vs prediction and cache. `.md`: 树与 MARS 的计算/并行. `done`; commit.

### Task 8: 11.8 — Bibliographic Notes (→ `done`)
`.py` prints refs (Breiman CART; Morgan & Sonquist AID; Friedman MARS/PRIM). `.md` brief. `done`; commit.

### Task 9: 11.9 — Exercises (→ `done`)
`.py`: tree vs linear (or GAM) comparison battery; `.md` 3 Chinese exercises + solved note. `done`.

### Task 10: README refresh + full verification
- [ ] `python scripts/generate_docs.py --regenerate`
- [ ] `python -m pytest` (≥8) + `python scripts/smoke_test.py`
- [ ] confirm 11.2/11.4 → 🔬
- [ ] commit `docs: Phase 10 additive models & trees content`

## Self-Review
**Spec coverage:** all 9 ch11 sections authored; deep 11.2/11.4 notebooks; progress/README. ✓
**Placeholder scan:** every task has concrete `.py` + Chinese `.md`; nb builders `build_nb_112.py`, `build_nb_114.py`. ✓
**Type consistency:** 11.2/11.4 → `notebook`; others `done`; keep the `:` in 11.4 section key. ✓
**Engineer note:** careful with absolute paths; never run `main.py`; MARS/MH in 11.4 use hinge bases in a manual loop.