# ESL Linear Classification (Phase 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapter 06 (Linear Methods for Classification, 7 sections): runnable `.py` for every section and rich `.ipynb` notebooks for the two deep sections (6.3 LDA, 6.4 Logistic regression). Update `progress.yaml`/`README.md` to ✅/🔬.

**Architecture:** Established pattern — direct-write Chinese `.md` + runnable `.py`; build deep `.ipynb` via `nbformat`; flip `progress.yaml` statuses; rebuild README via `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib, nbformat.

## Global Constraints

- `.md` narrative in **Chinese**; technical terms/code English; math in LaTeX. `.md` begins with `# <Full English Title>（<中文名>）`.
- Every `.py` runs standalone headless: `import matplotlib; matplotlib.use("Agg")` before `pyplot`; `np.random.seed(0)`/`default_rng(0)`; `if __name__ == "__main__":`; figures saved as `<sectiondir>/<section>_<name>.png`. **Never run `main.py`.**
- `progress.yaml` keys canonical, e.g. `06_Linear_Methods_for_Classification/6.3_Linear_Discriminant_Analysis`.
- Statuses: `done` / `notebook`; deep this phase: **6.3, 6.4**.
- Suppress benign `RuntimeWarning`s in classification scripts.
- Font preamble for plots:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
```

## Standard 5-step task envelope (content per task below)

- [ ] 1. Write `<section>.py` 2. Write `<section>.md` 3. progress override 4. run `.py` 5. commit

---

### Task 1: 6.1 — Introduction (→ `done`)

**Step 1 `.py`:** roadmap.

```python
# 6.1_Introduction
"""第6章概述: 线性分类。"""
ROADMAP = [
    ("6.2", "指示矩阵的线性回归"),
    ("6.3", "线性判别分析 LDA"),
    ("6.4", "逻辑回归"),
    ("6.5", "分离超平面"),
]


def main() -> None:
    print("第6章: Linear Methods for Classification")
    for k, t in ROADMAP:
        print(f"  {k}: {t}")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** Chinese intro: 用线性决策边界 $\{x:\beta_0+x^T\beta=0\}$ 做分类，两类方法（判别型 vs 概率型），路线图。Steps 3–5: `6.1`→`done`, run, commit.

---

### Task 2: 6.2 — Linear Regression of an Indicator Matrix (→ `done`)

**Step 1 `.py`:** encode 3-class Iris into indicator matrix, fit linear regression per column, predict by argmax.

```python
# 6.2_Linear_Regression_of_an_Indicator_Matrix
"""用指示矩阵做线性回归分类: 对每列回归再取 argmax。"""
import warnings
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_iris(return_X_y=True)
    Y0 = (y[:, None] == np.arange(3)).astype(float)  # K 个0-1 指示
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y0, stratify=y, test_size=0.3, random_state=0)
    model = LinearRegression().fit(Xtr, Ytr)
    pred = np.argmax(model.predict(Xte), axis=1)
    acc = np.mean(pred == np.argmax(Yte, axis=1))
    print(f"指示矩阵线性回归 分类准确率 = {acc:.3f}")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** 线性回归把响应编码为 $K$ 个指示变量，分类取 $\arg\max$；能在线性可分下合理，但可能产生"压掩效应"（某些类被其他类掩盖）。

**Steps 3–5:** `done`; commit `content: 6.2 indicator matrix regression`.

---

### Task 3: 6.3 — Linear Discriminant Analysis, **DEEP → `notebook`**

**Step 1 `.py`:** LDA decision boundaries on iris + report discriminant function.

```python
# 6.3_Linear_Discriminant_Analysis
"""线性判别分析 LDA: 决策边界与判别函数。"""
import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    iris = load_iris()
    X = iris.data[:, :2]  # 用两个特征便于可视化
    y = iris.target
    lda = LinearDiscriminantAnalysis().fit(X, y)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 300),
                         np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 300))
    Z = lda.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, Z, alpha=0.3, cmap="Pastel1")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", s=15)
    plt.title("LDA 决策区域 (鸢尾花前两特征)")
    plt.savefig("06_Linear_Methods_for_Classification/6.3_Linear_Discriminant_Analysis/6.3_LDA边界.png", dpi=110)
    print("训练准确率:", round(lda.score(X, y), 3))


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** 假设各类高斯、同协方差 $\Sigma$，贝叶斯规则 → 线性判别函数 $\delta_k(x)=x^T\Sigma^{-1}\mu_k-\tfrac12\mu_k^T\Sigma^{-1}\mu_k+\log\pi_k$；边界为线。提到正则化/降维。

**Steps 3–5:** notebook 6.3 (`build_nb_63.py`: 推导 + iris 可视化) → `6.3...: stub`→`notebook` → commit `content: 6.3 LDA (deep + notebook)`.

---

### Task 4: 6.4 — Logistic Regression, **DEEP → `notebook`**

**Step 1 `.py`:** logistic regression coefficients + decision boundary on two-class data.

```python
# 6.4_Logistic_Regression
"""逻辑回归: 拟合二分类并绘制决策边界。"""
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_blobs(n_samples=300, centers=2, cluster_std=2.0, random_state=0)
    clf = LogisticRegression().fit(X, y)
    gx, gy = np.meshgrid(np.linspace(X[:,0].min()-2, X[:,0].max()+2, 300),
                         np.linspace(X[:,1].min()-2, X[:,1].max()+2, 300))
    z = clf.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, z, alpha=0.3, cmap="bwr")
    plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", s=12)
    plt.title("逻辑回归决策边界")
    plt.savefig("06_Linear_Methods_for_Classification/6.4_Logistic_Regression/6.4_决策边界.png", dpi=110)
    print("系数(截距,beta1,beta2):", np.round(clf.intercept_[0],2), np.round(clf.coef_[0],2))


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** 逻辑函数 $\sigma(u)=1/(1+e^{-u})$，$\log\frac{p}{1-p}=\beta^T x$；用极大似然/IRLS 拟合。**二分情形简洁、稳定；多分类用多项逻辑回归。**

**Steps 3–5:** notebook `6.4` (`build_nb_64.py`: 多元逻辑回归 + 决策面) → `6.4`→`notebook`; commit `content: 6.4 logistic regression (deep + notebook)`.

---

### Task 5: 6.5 — Separating Hyperplanes (→ `done`)

**Step 1 `.py`:** perceptron / optimal separating line demo via `svm.LinearSVC` (max-margin) boundary.

```python
# 6.5_Separating_Hyperplanes
"""分隔超平面: 线性SVM(最大间隔)绘制边界。"""
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_blobs(n_samples=120, centers=2, cluster_std=1.2, random_state=0)
    clf = LinearSVC(C=1e6).fit(X, y)
    gx, gy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300))
    z = clf.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, z, alpha=0.25, cmap="PRGn")
    plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", s=12)
    plt.title("最大间隔分隔超平面")
    plt.savefig("06_Linear_Methods_for_Classification/6.5_Separating_Hyperplanes/6.5_boundary.png", dpi=110)
    print("分隔超平面系数:", np.round(clf.coef_[0], 3))


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** 判别型方法找分离面；Rosenblatt 感知机与最大间隔支持向量；不可分情形讨论 Hinge 损失。
**Steps 3–5:** `done`; commit `content: 6.5 separating hyperplanes`.

---

### Task 6: 6.6 — Bibliographic Notes (→ `done`)
`.py` prints refs (Fisher LDA; Rao; logistic—Berkson/Cox; SVM—Vapnik). Short `.md` guidance. `done`; commit.

### Task 7: 6.7 — Exercises (→ `done`)
`scikit-learn` exercisable: compare LDA vs logistic on iris (accuracy) + .md with 3 Chinese exercises and a solved example note. `done`; commit.

---

### Task 8: README refresh + full verification
- [ ] `python scripts/generate_docs.py --regenerate`
- [ ] `python -m pytest` (≥8) + `python scripts/smoke_test.py`
- [ ] confirm README: 6.3/6.4 as 🔬
- [ ] commit `docs: Phase 5 linear classification content`

## Self-Review

**Spec coverage (§10 Phase 5):** all 7 ch6 sections authored; two deep notebooks built (6.3, 6.4); progress/README updated; figures per §6. ✓
**Placeholder scan:** every task has full `.py` + Chinese `.md`; notebooks built by `build_nb_{3,4}.py` inline. ✓
**Type consistency:** `6.3`/`6.4` → `notebook`; others `done`; `.ipynb` names match section dirs. ✓
**Engineer reminder:** pay attention to absolute paths when writing `.md` under `../../config/progress.yaml`; never run `main.py`.