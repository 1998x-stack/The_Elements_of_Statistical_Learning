# ESL Neural Networks (Phase 12) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapter 13 (Neural Networks, 12 sections): runnable `.py` for every section and a rich `.ipynb` notebook for the one deep section (13.3 Neural Networks). Update `progress.yaml`/`README.md`.

**Architecture:** Established pattern — direct-write Chinese `.md` + runnable `.py`; build the deep `.ipynb` via `nbformat`; flip `progress.yaml`; rebuild README with `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib, nbformat. (NN via `sklearn.neural_network.MLPRegressor/MLPClassifier` — no torch needed.)

## Global Constraints

- `.md` narrative in **Chinese**; code/terms English; math in LaTeX. `.md` begins with `# <Full English Title>（<中文名>）`.
- `.py` runs standalone headless (`matplotlib.use("Agg")`), seeded, `if __name__ == "__main__":`, figures saved. **Never run `main.py`.**
- `progress.yaml` keys canonical: `13_Neural_Networks/<section>`.
- Statuses `done`/`notebook`; deep this phase: **13.3 only** (config confirms). Note section keys with `?`? none; `:` in `13.6_Example:_Simulated_Data`, `13.7_Example:_ZIP_Code_Data` — preserve.
- Suppress benign `RuntimeWarning`s; `MLP` can be slow — keep hidden sizes small and `max_iter` modest.
- Font preamble for plotting scripts.

## Standard 5-step envelope

- [ ] 1. `.py` 2. `.md` 3. progress 4. run 5. commit

---

### Task 1: 13.1 — Introduction (→ `done`)
`.py` roadmap. `.md`: NN 作为投影寻踪+平滑、通用逼近。`13.1...: stub`→`done`; commit.

### Task 2: 13.2 — Projection Pursuit Regression (→ `done`)
`.py`: 用张量/单指数演示 PPR 思想 (fitness of additive ridge). `.md`: 投影方向、指数函数。`done`; commit.

### Task 3: 13.3 — Neural Networks, **DEEP → notebook** (→ `notebook`)
`.py`: 用 `MLPRegressor`/`MLPClassifier` 在 2D 合成(classify) 上 fit 并画决策面 + 报告。`.md`: 单隐层网络 $f_k=\sigma(\beta_{0k}+\sum w_{jk} g_j(x))$, 激活、表达力。 build nb `build_nb_133.py`. `13.3...: stub`→`notebook`; commit.

### Task 4: 13.4 — Fitting Neural Networks (→ `done`)
`.py`: MLP 训练时观察 loss 曲线随迭代(epochs) 下降。`.md`: 反向传播误差、梯度下降、weight decay、误差面非凸。`done`; commit.

### Task 5: 13.5 — Some Issues in Training NNs (→ `done`)
`.py`: 学习率/正则(alpha)对 MLP 测试误差的影响; 标准化输入。`.md`: 缩放、初始化、过拟合控制、早停。`done`; commit.

### Task 6: 13.6 — Example: Simulated Data (→ `done`)
`.py`: 精度对 MLP 在合成二类上的拟合与决策面。`.md`: 小示例、激活层数选择。`done`; commit.

### Task 7: 13.7 — Example: ZIP Code Data (→ `done`)
`.py`: 用 digits（代理）MLP 分类并打印准确率。`.md`: 手写数字、把多维输入、softmax。`done`; commit.

### Task 8: 13.8 — Discussion (→ `done`)
`.py`: 比较浅 MLP vs 核对 MLP 在 sklearn digits。`.md`: 结构选择、泛化、对维度相关调整。`done`; commit.

### Task 9: 13.9 — Bayesian Neural Nets & NIPS 2003 (→ `done`)
`.py`: 显示 NN 输出不确定性/平均化的简单演示(或打印概念)。`.md` 贝叶斯神经网络。`done`; commit.

### Task 10: 13.10 — Computational Considerations (→ `done`)
`.py`: MLP 拟合耗时 vs 数据规模. `.md`: 训练成本、批处理、GPU。`done`; commit.

### Task 11: 13.11 — Bibliographic Notes (→ `done`)
`.py` prints refs (Rumelhart backprop; LeCun; Cybenko universal approx). `.md` brief. `done`.

### Task 12: 13.12 — Exercises (→ `done`)
`.py`: varying hidden units/epochs vs CV error exercisable. `.md` 3 Chinese exercises + solved note. `done`.

### Task 13: README refresh + full verification
- [ ] `python scripts/generate_docs.py --regenerate`
- [ ] `python -m pytest` (≥8) + `python scripts/smoke_test.py`
- [ ] confirm 13.3 → 🔬
- [ ] commit `docs: Phase 12 neural networks content`

## Self-Review
**Spec coverage:** all 12 ch13 sections authored; deep 13.3 notebook; progress/README. ✓
**Placeholder scan:** every task has concrete `.py` + Chinese `.md`; nb `build_nb_133.py`. ✓
**Type consistency:** 13.3 → notebook; others `done`. ✓
**Engineer note:** careful with `:` in 13.6/13.7 section keys; never run `main.py`; keep `MLP` cheap for smoke (small hidden, few iter).