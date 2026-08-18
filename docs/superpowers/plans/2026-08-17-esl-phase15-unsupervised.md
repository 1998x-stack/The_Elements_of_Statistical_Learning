# ESL Unsupervised Learning (Phase 15) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapter 16 (Unsupervised Learning, 12 sections): runnable `.py` for every section and rich `.ipynb` notebooks for the two deep sections (16.3 Cluster Analysis, 16.5 Principal Components). Update `progress.yaml`/`README.md`.

**Architecture:** Established pattern — direct-write Chinese `.md` + runnable `.py`; build deep `.ipynb` via `nbformat`; flip `progress.yaml`; rebuild README with `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib, nbformat.

## Global Constraints

- `.md` narrative in **Chinese**; code/terms English; math in LaTeX. `.md` begins with `# <Full English Title>（<中文名>）`.
- `.py` runs standalone headless (`matplotlib.use("Agg")`), seeded, `if __name__ == "__main__":`, figures saved. **Never run `main.py`.**
- `progress.yaml` keys canonical: `16_Unsupervised_Learning/<section>`.
- Statuses `done`/`notebook`; deep this phase: **16.3, 16.5**.
- Suppress benign `RuntimeWarning`s.
- Font preamble for plotting scripts.

## Standard 5-step envelope

- [ ] 1. `.py` 2. `.md` 3. progress 4. run 5. commit

---

### Task 1: 16.1 — Introduction (→ `done`)
`.py` roadmap. `.md`: 无监督目标、降维/聚类/关联。`16.1...: stub`→`done`; commit.

### Task 2: 16.2 — Association Rules (→ `done`)
`.py`: 简单 apriori 思想生成一组项集并算提升度。`.md`: 购物篮、支持/置信/提升。`done`; commit.

### Task 3: 16.3 — Cluster Analysis, **DEEP → notebook** (→ `notebook`)
`.py`: k-means（`KMeans`）+ 层次（`AgglomerativeClustering`）在合成数据, 对比。`.md`: 划分/层次/基于模型聚类、距离选择。 build nb `build_nb_163.py`. `16.3...: stub`→`notebook`; commit.

### Task 4: 16.4 — Self-Organizing Maps (→ `done`)
`.py`: 用 `MiniSom`? 可能缺库 → 用简化/示意或对 2D 网格做原型聚类的代理。`.md`: SOM、拓扑保持。`done`; commit. (用 sk-kmeans 近似每个网格质心)

### Task 5: 16.5 — Principal Components, **DEEP → notebook** (→ `notebook`)
`.py`: `sklearn PCA` 在 iris 降维到 2D 并散点图, 解释方差占比。  `.md`: PCA 的谱分解、降维、主曲线。 build nb `build_nb_165.py`. `16.5→notebook`; commit.

### Task 6: 16.6 — Non-negative Matrix Factorization (→ `done`)
`.py`: `NMF` 在 digits 上因子分解/部分表示。`.md`: NMF、非负散点/特征组。`done`; commit.

### Task 7: 16.7 — ICA & Exploratory Projection Pursuit (→ `done`)
`.py`: `FastICA` 在混合独立信号上恢复. `.md`: ICA 假设非高斯、盲源分离。`16.7→done`; commit.

### Task 8: 16.8 — Multidimensional Scaling (→ `done`)
`.py`: `MDS` 对距离阵嵌入 2D. `.md`: MDS、保留距离/k角度。`done`; commit.

### Task 9: 16.9 — Nonlinear Dimension Reduction & Local MDS (→ `done`)
`.py`: Isomap/LocallyLinearEmbedding(SLE)/对比较. `.md`: 流形学习、局部距离. `.done`; commit.

### Task 10: 16.10 — The Google PageRank Algorithm (→ `done`)
`.py`: 用 mini 邻接矩阵迭代 PageRank。`.md`: 随机游走、PageRanks。`done`; commit.

### Task 11: 16.11 — Bibliographic Notes (→ `done`)
`.py` prints refs (Hartigan & Wong kmeans; Friedman clustering; Scholkopf PCA; Cover). `.md` brief. `done`; commit.

### Task 12: 16.12 — Exercises (→ `done`)
`.py`: k 的选择与初始化在 k-means 的作用（演示 k）。`.md`: 3 个中文习题+参考。`done`.

### Task 13: README refresh + full verification
- [ ] `python scripts/generate_docs.py --regenerate`
- [ ] `python -m pytest` (≥8) + `python scripts/smoke_test.py`
- [ ] confirm 16.3/16.5 → 🔬
- [ ] commit `docs: Phase 15 unsupervised learning content`

## Self-Review
**Spec coverage:** all 12 ch16 sections authored; deep 16.3/16.5 notebooks; progress/README. ✓
**Placeholder scan:** per-task `.py` + Chinese `.md`; nb builders `build_nb_{163,165}.py`. ✓
**Type consistency:** 16.3/16.5 → notebook; others → done. ✓
**Engineer note:** never run `main.py`; careful with absolute paths; `SOM` may need a manual/approx demo if no package.