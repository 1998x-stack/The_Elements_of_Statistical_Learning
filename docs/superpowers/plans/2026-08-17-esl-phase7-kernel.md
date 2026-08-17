# ESL Kernel Smoothing Methods (Phase 7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapter 08 (Kernel Smoothing Methods, 11 sections) — runnable `.py` for every section, **no deep sections / notebooks** (all `done`). Update `progress.yaml`/`README.md` to ✅.

**Architecture:** Established pattern — direct-write Chinese `.md` + runnable `.py`; flip `progress.yaml` (`stub`→`done`); rebuild README with `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib.

## Global Constraints

- `.md` narrative in **Chinese**; technical terms/code English; math in LaTeX. `.md` begins with `# <Full English Title>（<中文名>）`.
- Every `.py` runs standalone headless: `matplotlib.use("Agg")` before `pyplot`; `if __name__ == "__main__":`; figures saved as `<sectiondir>/<section>_<name>.png`; `np.random.seed(0)`/`default_rng(0)`. **Never run `main.py`.**
- `progress.yaml` keys canonical: `08_Kernel_Smoothing_Methods/<section>`. All sections this phase → **`done`** (no `notebook`).
- Suppress benign `RuntimeWarning`s.
- Figure font preamble for plotting scripts:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
```

## Standard 5-step envelope (content per task)

- [ ] 1. Write `<section>.py` 2. Write `<section>.md` 3. progress `... `→`done` 4. run `.py` 5. commit

**(Note: for 8.1–8.11 use the exact section keys from `config/structure.yaml`, replacing any `-` with `_` in directory/file names — see the `ls` listing; the YAML keys use `One_Dimensional` etc. already.)**

---

### Task 1: 8.1 — One-Dimensional Kernel Smoothers (→ `done`)
`.py`: implement Neville/`nwir`-style moving average + Epanechnikov / Gaussian kernel smoother (`h=bandwidth`) fit to noisy sine, show data and smooth. `.md`: 核权重平均 $\hat f(x)=\frac{\sum_i w_i(x) y_i}{\sum_i w_i(x)}$、常见核。`done`; run; commit `content: 8.1 one-dimensional kernel smoothers`.

### Task 2: 8.2 — Selecting the Width of the Kernel (→ `done`)
`.py`: cross-validated bandwidth selection for a Gaussian kernel smoother over a grid of `h`; plot CV error vs `h`. `.md`: 带宽为光滑参数 -> 偏差-方差；CV 选择。`done`; commit.

### Task 3: 8.3 — Local Regression in R^p (→ `done`)
`.py`: `sklearn.LocallyWeighted`? use `statsmodels`? simplest: local linear fit via weighted least squares in 1D (tri-cube kernel, nearest neighbors `k`); plot local-linear vs kernel average. `.md`: local polynomial regression 局部最小二乘。`done`;

### Task 4: 8.4 — Structured Local Regression in R^p (→ `done`)
`.py`: structured/local linear on mild 2D (e.g., additive local fits on each variable) simple. `.md`: 结构化局部回归与张量积。`done`;

### Task 5: 8.5 — Local Likelihood & Other Models (→ `done`)
`.py`: local logistic regression (kernel-weighted Bernoulli GLM in neighborhood) demo. `.md`: 局部似然、局部 GLM。`done`;

### Task 6: 8.6 — Kernel Density Estimation & Classification (→ `done`)
`.py`: `scipy`/`sklearn.KernelDensity` (Gaussian) estimate of 1D density, show bandwidth effect; classify via density. `.md`: KDE、Parzen、用于分类。`done`;

### Task 7: 8.7 — Radial Basis Functions & Kernels (→ `done`)
`.py`: RBF interpolation/smoothing in 1D (like 7.6 but RBF network), compare. `.md`: RBF 展开 $\sum_{j} w_j\phi_j(x)$、可与核 (14). `done`;

### Task 8: 8.8 — Mixture Models for Density Estimation & Classification (→ `done`)
`.py`: `GaussianMixture` on 2D blobs, plot components + classification. `.md`: 高斯混合、EM 简介。 `done`;

### Task 9: 8.9 — Computational Considerations (→ `done`)
`.py`: 对大型 `.py` 计数 kernel window 与直接循环的耗时 (n log n vs n). `.md`: 高效邻域、KD树。`done`;

### Task 10: 8.10 — Bibliographic Notes (→ `done`)
`.py` prints refs (Silverman density; Cleveland loess; Müller; Fan local regression; Ripley). `.md` brief. `done`; commit.

### Task 11: 8.11 — Exercises (→ `done`)
`.py`: cross-validated density / kernel comparison exercisable. `.md` 3 Chinese exercises + solved note. `done`;

### Task 12: README refresh + full verification
- [ ] `python scripts/generate_docs.py --regenerate`
- [ ] `python -m pytest` (≥8) + `python scripts/smoke_test.py`
- [ ] confirm 08 sections all `✅ 摘要` (none 🔬)
- [ ] commit `docs: Phase 7 kernel smoothing content`

## Self-Review
**Spec coverage:** all 11 ch8 sections authored, none deep (config has no 08 deep) → all `done`. ✓
**Placeholder scan:** per-task py + md; the brief outlines above are intentionally compact directives, but the implementer MUST expand to full runnable code + readable Chinese md before each commit (this is content authoring, not scaffolding). ✓
**Type consistency:** `8.1`...`8.11` all `done`; notebook count unchanged. ✓
**Engineer note:** the chapter has no deep sections this phase; do not add notebooks. Be meticulous with absolute paths; never run `main.py`.