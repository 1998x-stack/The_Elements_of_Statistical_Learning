# ESL Boosting & Additive Trees (Phase 11) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapter 12 (Boosting & Additive Trees, 16 sections): runnable `.py` for every section and rich `.ipynb` notebooks for the three deep sections (12.1 Boosting Methods, 12.4 Exponential Loss & AdaBoost, 12.10 Numerical Optimization via Gradient Boosting). Update `progress.yaml`/`README.md`.

**Architecture:** Established pattern — direct-write Chinese `.md` + runnable `.py`; build deep `.ipynb` via `nbformat`; flip `progress.yaml`; rebuild README with `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib, nbformat.

## Global Constraints

- `.md` narrative in **Chinese**; code/terms English; math in LaTeX. `.md` begins with `# <Full English Title>（<中文名>）`.
- `.py` runs standalone headless (`matplotlib.use("Agg")`), seeded, `if __name__ == "__main__":`, figures saved. **Never run `main.py`.**
- `progress.yaml` keys canonical: `12_Boosting_and_Additive_Trees/<section>`.
- Statuses `done`/`notebook`; deep this phase: **12.1, 12.4, 12.10**. Some section keys contain `?` (12.5 `Why_Exponential_Loss?`) — preserve exactly.
- Suppress benign `RuntimeWarning`s.
- Font preamble for plotting scripts.

## Standard 5-step envelope

- [ ] 1. `.py` 2. `.md` 3. progress 4. run 5. commit

---

### Task 1: 12.1 — Boosting Methods, **DEEP → notebook** (→ `notebook`)
`.py`: 手写基础 AdaBoost（指数损失，样本权重）在 `make_moons` 上, 打印每轮训练/测试误差。`.md`: 从“弱学习器”到强学习器、AdaBoost 伪代码基础。 build nb `build_nb_121.py`. `12.1...: stub`→`notebook`; commit.

### Task 2: 12.2 — Boosting Fits an Additive Model (→ `done`)
`.py`: 展示 AdaBoost 为加性模型（指数损失的 stagewise）演示累加弱学习器。`.md`: $f(x)=\sum_m\alpha_m G_m(x)$, 加性解释。`done`; commit.

### Task 3: 12.3 — Forward Stagewise Additive Modeling (→ `done`)
`.py`: stagewise additive 对平方损失逐步加小的基学习器。`.md`: 前向分步加性建模(FSAM)。`done`; commit.

### Task 4: 12.4 — Exponential Loss & AdaBoost, **DEEP → neb** (→ `notebook`)
`.py`: 手写 AdaBoost（指数损失 + 样本权重）在二维分类上运行, 画每轮加权误差。`.md`: AdaBoost 算法、样本权重、指数损失 ⇔ FSAM 解。`12.4→`notebook`; build nb `build_nb_124.py`; commit.

### Task 5: 12.5 — Why Exponential Loss? (→ `done`)
`.py`: 对比指数损失与 0-1 损失, 演示其数值近似。`.md`: 指数损失台是 0-1/平滑上界, 但受离群点影响。`12.5...Why_Exponential_Loss?`→`done`; commit.

### Task 6: 12.6 — Loss Functions & Robustness (→ `done`)
`.py`: 对比平方、绝对、Huber、logit 损失在含离群点回归上。`.md`: 各损失鲁棒性与稀疏。`done`; commit.

### Task 7: 12.7 — Off-the-Shelf Procedures for Data Mining (→ `done`)
`.py`: 一行 sklearn `GradientBoostingClassifier` 在 iris 上演示 off-the-shelf。`.md`: 现成 boosting 工具、适用建议。`done`; commit.

### Task 8: 12.8 — Example: Spam Data (→ `done`)
`.py`: 对 spam-ish 合成/低维演示 boosting 分类器误差. `.md`: 综述垃圾邮件例。`done`; commit.

### Task 9: 12.9 — Boosting Trees (→ `done`)
`.py`: boosting 树(staged decision trees)误差随 add# 的曲线, 与单树对比。`.md`: 树作为基弱学习器，残差引导。`done`; commit.

### Task 10: 12.10 — Numerical Optimization via Gradient Boosting, **DEEP → neb** (→ `notebook`)
`.py`: 自实现 gradient boosting 回归: 每轮对负梯度拟合小树加 η。`.md`: GBM=在函数空间数值优化、学习率、树深。`12.10...: stub`→`notebook`; build nb `nb_1210.py`; commit.

### Task 11: 12.11 — Right-Sized Trees for Boosting (→ `done`)
`.py`: 变树深度(2,4,8)对 boosting 测试误差影响. `.md`: 树大小=复杂度参数，一般小树。`done`; commit.

### Task 12: 12.12 — Regularization (→ `done`)
`.py`: 改变学习率 η 与 M 看误差-早停. `.md`: 收缩 η、M 选择、交叉验证。`done`; commit.

### Task 13: 12.13 — Interpretation (→ `done`)
`.py`: 特征相对重要性/相对贡献条形图。`.md`: 相对重要性、偏依赖图。`done`; commit.

### Task 14: 12.14 — Illustrations (→ `done`)
`.py`: 2D boosting 决策面图对比单树。`.md`: 可视化示例。`done`; commit.

### Task 15: 12.15 — Bibliographic Notes (→ `done`)
`.py` prints refs (Freund & Schapire AdaBoost; Breiman Arcing; Friedman). `.md` brief. `done`; commit.

### Task 16: 12.16 — Exercises (→ `done`)
`.py`: GBM depth/learning-rate CV comparison exercisable; `.md` 3 Chinese exercises + solved note. `done`.

### Task 17: README refresh + full verification
- [ ] `python scripts/generate_docs.py --regenerate`
- [ ] `python -m pytest` (≥8) + `python scripts/smoke_test.py`
- [ ] confirm 12.1/12.4/12.10 → 🔬
- [ ] commit `docs: Phase 11 boosting content`

## Self-Review
**Spec coverage:** all 16 ch12 sections authored; deep 12.1/12.4/12.10 notebooks; progress/README. ✓
**Placeholder scan:** every task has concrete `.py` + Chinese `.md`; nb builders for the 3 deep. ✓
**Type consistency:** `?`-named 12.5 preserved; deep set = 12.1,12.4,12.10. ✓
**Engineer note:** 绝不运行 `main.py`；注意 `?` 命名的 12.5；数字路径精确。