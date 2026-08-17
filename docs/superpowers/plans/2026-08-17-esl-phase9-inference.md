# ESL Model Inference & Averaging (Phase 9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapter 10 (Model Inference & Averaging, 11 sections): runnable `.py` for every section and rich `.ipynb` notebooks for the two deep sections (10.5 EM, 10.7 Bagging). Update `progress.yaml`/`README.md`.

**Architecture:** Established pattern — direct-write Chinese `.md` + runnable `.py`; build deep `.ipynb` via `nbformat`; flip `progress.yaml`; rebuild README with `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib, nbformat.

## Global Constraints

- `.md` narrative in **Chinese**; code/terms English; math in LaTeX. `.md` begins with `# <Full English Title>（<中文名>）`.
- `.py` runs standalone headless (`matplotlib.use("Agg")`), seeded, `if __name__ == "__main__":`, figures saved. **Never run `main.py`.**
- `progress.yaml` keys canonical: `10_Model_Inference_and_Averaging/<section>`.
- Statuses `done`/`notebook`; deep this phase: **10.5** (The EM Algorithm), **10.7** (Bagging).
- Suppress benign `RuntimeWarning`s.
- Figure font preamble for plotting scripts.

## Standard 5-step envelope

- [ ] 1. `.py` 2. `.md` 3. progress 4. run 5. commit

---

### Task 1: 10.1 — Introduction (→ `done`)
`.py` roadmap. `.md`: 推断 (MLE/贝叶斯/自举) 与平均化 (bagging/stacking)。`10.1...: stub`→`done`; commit.

### Task 2: 10.2 — The Bootstrap & Maximum Likelihood Methods (→ `done`)
`.py`: bootstrap CI for a model coefficient against MLE-based CI. `.md`: 自举与 MLE 的一致性; delta method。`done`; commit.

### Task 3: 10.3 — Bayesian Methods (→ `done`)
`.py`: 单片简单高斯后验(共轭)演示, 均值后验 vs MLE。`.md`: 先验-似然后验、共轭、MAP。`done`; commit.

### Task 4: 10.4 — Relationship Between Bootstrap & Bayesian (→ `done`)
`.py`: 用自举近似后验分布(作为一个粗略的非参数贝叶斯)。 `.md`: bootstrap 与贝叶斯推断的联系与非参数后验。`done`; commit.

### Task 5: 10.5 — The EM Algorithm, **DEEP → `notebook`** (→ `notebook`)
`.py`: Gaussian mixture EM 逐迭代(手写合成) 打印 log-likelihood递增。`.md`: E-step/Q-step/fisher; log-likelihood不减、收敛。 `.md`. Build nb `build_nb_105.py` (分步实现 EM). `10.5...: stub`→`notebook`; commit.
*(notebook filename `scripts/build_nb_105.py`.)*

### Task 6: 10.6 — MCMC for Sampling from the Posterior (→ `done`)
`.py`: Metropolis-Hastings 采简单后验(高斯)演示采样路径/直方图。`.md`: MCMC、MH、Gibbs。 `10.6`→`done`; commit.

### Task 7: 10.7 — Bagging, **DEEP → `notebook`** (→ `notebook`)
`.py`: 用多个重采样拟合决策树并对预测平均(bagging) 与单棵树对比。用 sklearn RandomForest 前先手写 bagging。`.md`: 平均化降方差、缓解过拟合。 build nb `build_nb_107.py`. `10.7...: stub`→`notebook`； commit.

### Task 8: 10.8 — Model Averaging & Stacking (→ `done`)
`.py`: 线性回归的贝叶斯模型平均与 stacking(元层) 简单演示。`.md`: BMA、stacking。`10.8`→`done`; commit.

### Task 9: 10.9 — Stochastic Search: Bumping (→ `done`)
`.py`: 用多次随机重启+留出选择(Bumping) 演示, 对比择优。`.md`: bumping、留意“运气”。`10.9`→`done`; commit.

### Task 10: 10.10 — Bibliographic Notes (→ `done`)
`.py` prints refs (Efron bootstrap; Breiman bagging; Radford & Neal MCMC; Dempster EM; Wolpert stacking). `.md` brief. `done`;

### Task 11: 10.11 — Exercises (→ `done`)
`.py`: bagging vs single tree exercisable; `.md` 3 Chinese exercises + solved note. `done`.

### Task 12: README refresh + full verification
- [ ] `python scripts/generate_docs.py --regenerate`
- [ ] `python -m pytest` (≥8) + `python scripts/smoke_test.py`
- [ ] confirm 10.5/10.7 → 🔬
- [ ] commit `docs: Phase 9 model inference & averaging content`

## Self-Review
**Spec coverage:** all 11 ch10 sections; deep 10.5/10.7 notebooks; progress/README. ✓
**Placeholder scan:** every task has a concrete `.py` and Chinese `.md` (tasks 5 & 7 also fully spec the notebook). ✓
**Type consistency:** 10.5/10.7 → `notebook`; others `done`. ✓
**Engineer note:** careful with absolute **paths** and with the `build_nb_*`.py names (`105`,`107`) used for notebook generation; never run `main.py`.