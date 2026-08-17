# ESL Model Assessment & Selection (Phase 8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapter 09 (Model Assessment & Selection, 14 sections): runnable `.py` for every section and rich `.ipynb` notebooks for the two deep sections (9.10 Cross-Validation, 9.11 Bootstrap). Update `progress.yaml`/`README.md`.

**Architecture:** Established pattern — direct-write Chinese `.md` + runnable `.py`; build deep `.ipynb` via `nbformat`; flip `progress.yaml`; rebuild README with `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib, nbformat.

## Global Constraints

- `.md` narrative in **Chinese**; technical terms/code English; math in LaTeX. `.md` begins with `# <Full English Title>（<中文名>）`.
- Every `.py` runs standalone headless: `matplotlib.use("Agg")` before `pyplot`; `if __name__ == "__main__":`; figures saved as `<sectiondir>/<section>_<name>.png`; seeded RNG. **Never run `main.py`.**
- `progress.yaml` keys canonical, e.g. `09_Model_Assessment_and_Selection/9.10_Cross_Validation`. Note en-dashes in section names: `9.3_The_Bias–Variance_Decomposition`, `9.9_Vapnik–Chervonenkis_Dimension` (U+2013, do not normalize).
- Statuses `done`/`notebook`; deep this phase: **9.10, 9.11**.
- Suppress benign `RuntimeWarning`s.
- Figure font preamble for plotting scripts:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
```

## Standard 5-step envelope (per task)

- [ ] 1. Write `<section>.py` 2. `<section>.md` 3. progress override 4. run `.py` 5. commit

---

### Task 1: 9.1 — Introduction (→ `done`)
`.py` prints roadmap. `.md`: 区分**训练误差 / 泛化(测试)误差**，模型选择的目标。`done`; commit.

### Task 2: 9.2 — Bias, Variance & Model Complexity (→ `done`)
`.py`: for polynomial fits of increasing degree plot in-sample vs hold-out error (U-shape). `.md`: 复杂度↔偏差-方差。`done`; commit.

### Task 3: 9.3 — The Bias–Variance Decomposition (→ `done`)
`.py`: simulate bias/variance/MSE for kNN at multiple k (like 4.9) printing decomposition; note en-dash in name. `.md`: E[(y0−f̂)²]=Bias²+Var+σ²。`done`; commit.

### Task 4: 9.4 — Optimism of the Training Error Rate (→ `done`)
`.py`: show training error is optimistic (lower) than test error; measure "optimism". `.md`: optimism = E[err_train − err_test]; `done`; commit.

### Task 5: 9.5 — Estimates of In-Sample Prediction Error (→ `done`)
`.py`: compare AIC/Cp-style estimate vs test error for a linear regression on simulated data. `.md`: 训练误差估计: R2/Cp、AIC。`done`; commit.

### Task 6: 9.6 — The Effective Number of Parameters (→ `done`)
`.py`: ridge `df=tr(Sλ)` changes with λ; print df vs λ. `.md`: 有效自由度如岭/Smoother。 `done`; commit.

### Task 7: 9.7 — The Bayesian Approach & BIC (→ `done`)
`.py`: Gaussian likelihood + BIC for a small linear model, print BIC for different model sizes. `.md`: 后验一致性、BIC≈−2logL+df·log n。`done`; commit.

### Task 8: 9.8 — Minimum Description Length (→ `done`)
`.py`: piecewise-constant description-length demo (encode fit + residual cost) vs complexity. `.md`: MDL 与编码长度。`done`; commit.

### Task 9: 9.9 — Vapnik–Chervonenkis Dimension (→ `done`)
`.py`: empirically shatter some 1D threshold classifiers to illustrate VC growth (number of separable labelings as function of n). `.md`: VC 维度含义与示例。 `done`; commit.

### Task 10: 9.10 — Cross-Validation, **DEEP → `notebook`** (→ `notebook`)
`.py`: K-fold CV (K=5,10) + LOOCV compare training/val curves on a model-complexity sweep; plot. `.md`: K-fold、bias/variance of CV、LOOCV。Build nb `build_nb_910.py`. `9.10...: stub`→`notebook`; commit.

### Task 11: 9.11 — Bootstrap Methods, **DEEP → `notebook`** (→ `notebook`)
`.py`: parametric/ordinary bootstrap to estimate standard error of a statistic (e.g., median or linear coefficient) + "bootstrap resample" demo. `.md`: bootstrap 原理 + bagging前置。 Build nb `build_nb_911.py`. `notebook`; commit.

### Task 12: 9.12 — Conditional or Expected Test Error? (→ `done`)
`.py`: compare conditional (fixed training) vs expected test error empirically. `.md`: 有条件的 vs 期望测试误差。 `done`; commit.

### Task 13: 9.13 — Bibliographic Notes (→ `done`)
`.py` prints refs (Efron—bootstrap; Stone—CV; Akaike; Schwarz; Vapnik). `.md` brief; `done`; commit.

### Task 14: 9.14 — Exercises (→ `done`)
`.py`: CV + bootstrap comparison exercisable on a dataset. `.md` 3 Chinese exercises + solved note. `done`; commit.

### Task 15: README refresh + full verification
- [ ] `python scripts/generate_docs.py --regenerate`
- [ ] `python -m pytest` (≥8) + `python scripts/smoke_test.py`
- [ ] confirm 9.10/9.11 as 🔬
- [ ] commit `docs: Phase 8 model assessment & selection content`

## Self-Review
**Spec coverage:** all 14 ch9 sections authored; deep 9.10/9.11 notebooks; progress/README. ✓
**Placeholder scan:** per-task `.py` + Chinese `.md`; notebooks via `build_nb_{910,911}.py`. ✓
**Type consistency:** 9.10/9.11 → `notebook`; others `done`; keep en-dash in 9.3/9.9 names. ✓
**Engineer note:** en-dashes in 9.3 & 9.9 must be preserved; careful with absolute paths; never run `main.py`.