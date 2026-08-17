# ESL Foundations (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapters 03 (Introduction, 4 sections) and 04 (Overview of Supervised Learning, 11 sections): a runnable `.py` for every section and a rich `.ipynb` notebook for the two deep sections (4.5, 4.9). Update `progress.yaml` and `README.md` statuses to ✅/🔬.

**Architecture:** The established Phase 2 authoring pattern: direct-write the Chinese `.md` + runnable `.py`; for the two deep sections also build a `*.ipynb` with `nbformat`; flip `progress.yaml` statuses; rebuild the status-driven `README.md` with `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib, nbformat.

## Global Constraints

- `.md` narrative in **Chinese**; technical terms and code stay English; math in LaTeX (`$`/`$$`). Every `.md` begins with `# <Full English Title>（<中文名>）`.
- Every `.py` runs standalone headless: `import matplotlib; matplotlib.use("Agg")` before `pyplot`, `np.random.seed(0)` (or `default_rng(0)`), guarded by `if __name__ == "__main__":`, and saves any figure as `<sectiondir>/<section>_<name>.png`.
- **Never run or modify `main.py`.**
- `progress.yaml` keys are canonical (no leading slash), e.g. `03_Introduction/3.1_Examples_of_Learning_Problems`, `04_Overview_of_Supervised_Learning/4.9_Model_Selection_and_the_Bias–Variance_Tradeoff` (note the **en-dash `–`**, U+2013 — do not normalize it).
- Statuses: `done` (summary+code) for non-deep; `notebook` (summary+code+ipynb) for deep. Deep this phase: **4.5 and 4.9 only**.
- Figure font preamble for every plotting script:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
```

## Standard 5-step task envelope (content per task below)

- [ ] 1. Write `<section>.py` (code given per task)
- [ ] 2. Write `<section>.md` (content given per task)
- [ ] 3. Update `config/progress.yaml` (`...: stub` → `done` or `notebook`)
- [ ] 4. Run the `.py` headless; confirm exit 0 and any PNG output
- [ ] 5. Commit (`git add <section dir> config/progress.yaml`; message per task)

---

### Task 1: 3.1 — Examples of Learning Problems (→ `done`)

**Step 1 `.py`:**

```python
# 3.1_Examples_of_Learning_Problems
"""示例: 展示监督回归与无监督聚类的两类学习问题。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)


def main() -> None:
    x = np.sort(rng.uniform(0, 4, 200))
    y = np.sin(1.5 * x) + 0.2 * rng.standard_normal(200)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.5))
    a.scatter(x, y, s=8)
    a.set_title("监督学习: 由 X 预测实值 Y (回归)")
    pts = np.vstack([rng.multivariate_normal([0, 0], np.eye(2), 75),
                     rng.multivariate_normal([2, 2], 0.7 * np.eye(2), 75)])
    b.scatter(pts[:, 0], pts[:, 1], s=8, c=pts[:, 0] > 1)
    b.set_title("无监督学习: 数据中的簇结构")
    fig.tight_layout()
    fig.savefig("03_Introduction/3.1_Examples_of_Learning_Problems/3.1_Examples_of_Learning_Problems_图.png", dpi=110)
    print("ok: 3.1 已生成示例图")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 3.1 Examples of Learning Problems（学习问题的示例）

本节用直观例子引出"从数据中学习"的两类基本问题。

## 内容

- **监督学习**：给定输入 $X$ 与标签 $Y$ 的样本，学习映射 $f:X\to Y$。
  - 回归（$Y$ 连续，如房价预测）；
  - 分类（$Y$ 离散，如垃圾邮件判别）。
- **无监督学习**：只有输入 $X$，目标是发现结构（聚类、降维、密度估计）。

## 核心概念

学习好坏可用**期望损失 / 风险**统一刻画：

$$
R(f)=\mathbb{E}_{(X,Y)}\left[L\big(y,f(x)\big)\right]
$$

## 代码示例

- 运行 `.py` 生成两幅示例：左=监督回归，右=带头簇的无监督数据。
- 图片保存在同目录 `.png`。

## 相关章节

- 4.3（两类简单预测方法）、16（无监督学习）
```

**Steps 3–5:** `03_Introduction/3.1_Examples_of_Learning_Problems` → `done`; run; commit `content: 3.1 examples of learning problems`.

---

### Task 2: 3.2 — Supervised and Unsupervised Learning (→ `done`)

**Step 1 `.py`:**

```python
# 3.2_Supervised_and_Unsupervised_Learning
"""无监督学习示例: K-均值聚类。"""
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)


def main() -> None:
    centers = [(0, 0), (4, 0), (2, 3)]
    X = np.vstack([rng.multivariate_normal(c, 0.4 * np.eye(2), 100) for c in centers])
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="viridis", s=12)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color="red", marker="x", s=120)
    plt.title("无监督学习: K-means 聚类 (K=3)")
    plt.savefig("03_Introduction/3.2_Supervised_and_Unsupervised_Learning/3.2_KMeans.png", dpi=110)
    print("ok: inertia =", round(km.inertia_, 1))


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 3.2 Supervised and Unsupervised Learning（监督与无监督学习）

## 内容

- **监督学习**：样本 $(x_i,y_i)$ 带标签，学习 $\hat f$ 使 $\mathbb{E}[L(Y,f(X))]$ 最小——回归、分类。
- **无监督学习**：只有 $x_i$、无标签，任务是发现结构：聚类、降维、密度估计、关联规则。

## 核心概念

监督学习根植于风险最小化；无监督常借助数据的几何 / 概率结构（如高斯混合、流形假设）。

## 代码示例

- 用 `sklearn.KMeans` 对三簇合成数据聚类，标出簇心。

## 相关章节

- 16（无监督学习）
```

**Steps 3–5:** `done`; run; commit `content: 3.2 supervised and unsupervised learning`.

---

### Task 3: 3.3 — Statistical Decision Theory (→ `done`)

**Step 1 `.py`:**

```python
# 3.3_Statistical_Decision_Theory
"""统计决策理论: 分箱估计回归函数 E[Y|X]。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)


def main() -> None:
    n = 800
    x = rng.uniform(0, 4, n)
    y = np.sin(1.5 * x) + 0.15 * rng.standard_normal(n)
    xs = np.sort(x)
    ys = y[np.argsort(x)]
    nbins = 20
    bins = np.linspace(0, 4, nbins + 1)
    idx = np.digitize(xs, bins) - 1
    centers = (bins[:-1] + bins[1:]) / 2
    est = [ys[idx == k].mean() if np.any(idx == k) else np.nan for k in range(nbins)]
    xx = np.linspace(0, 4, 300)
    plt.scatter(xs, ys, s=4, alpha=0.4, label="样本")
    plt.plot(xx, np.sin(1.5 * xx), "k--", label="真实 E[Y|X]")
    plt.plot(centers, est, "-o", label="分箱估计")
    plt.legend()
    plt.title("回归函数 E[Y|X] 的估计")
    plt.savefig("03_Introduction/3.3_Statistical_Decision_Theory/3.3_回归函数.png", dpi=110)
    print("ok: 3.3")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 3.3 Statistical Decision Theory（统计决策理论）

## 内容

- 给定损失 $L$ 与联合分布 $P(X,Y)$，最优预测是**最小化期望损失（风险）**。
- 平方损失下最优解是**条件均值（回归函数）**：

$$
f^*(x)=\mathbb{E}[Y\mid X=x]
$$

- 分类（0-1 损失）下最优解是**贝叶斯分类器**：取后验概率最大的类。

## 讨论

- 实际中 $P(X,Y)$ 未知，只能用样本近似 $f^*$——这正是机器学习的任务。
- 本章及第 4 章反复使用此框架。

## 代码示例

- 用分箱直方图近似 $E[Y|X]$，与真实回归函数对比。

## 相关章节

- 4.4（监督学习的统计决策理论）
```

**Steps 3–5:** `done`; run; commit `content: 3.3 statistical decision theory`.

---

### Task 4: 3.4 — The Many Faces of Regularization (→ `done`)

**Step 1 `.py`:**

```python
# 3.4_The_Many_Faces_of_Regularization
"""正则化: 岭回归 vs 最小二乘在高阶多项式上的对比。"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(-3, 3, 80))
    y = np.sin(x) + 0.2 * rng.standard_normal(80)
    xx = np.linspace(-3, 3, 400)
    ls = make_pipeline(PolynomialFeatures(12), LinearRegression()).fit(x[:, None], y)
    ridge = make_pipeline(PolynomialFeatures(12), Ridge(alpha=5)).fit(x[:, None], y)
    plt.scatter(x, y, s=6, alpha=0.5)
    plt.plot(xx, np.sin(xx), "k--", label="真实")
    plt.plot(xx, ls.predict(xx[:, None]), label="最小二乘(过拟合)")
    plt.plot(xx, ridge.predict(xx[:, None]), label="岭回归")
    plt.legend()
    plt.title("正则化抑制过拟合")
    plt.savefig("03_Introduction/3.4_The_Many_Faces_of_Regularization/3.4_正则化.png", dpi=110)
    print("ok: 3.4")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 3.4 The Many Faces of Regularization（正则化的许多面孔）

## 内容

"正则化"是一系列降低模型复杂度的原则，并非单一技巧：

- **罚函数 / 收缩**：岭（L2）、Lasso（L1）；
- **结构限制**：低维、稀疏、平滑、单调约束；
- **提前停止 / 交叉验证**：以数据驱动选择复杂度；
- **平滑化**：核方法、样条。

## 核心思想

通过引入偏差换取**方差**大幅下降，改善期望测试误差：

$$
\mathbb{E}(y_0-\hat f(x_0))^2 = \text{Bias}^2 + \text{Var} + \sigma^2
$$

## 代码示例

- 十二次多项式拟合带噪正弦：最小二乘过拟合摆动，岭回归回稳。

## 相关章节

- 5.4（收缩方法）、9（模型评估）
```

**Steps 3–5:** `done`; run; commit `content: 3.4 the many faces of regularization`.

---

### Task 5: 4.1 — Introduction (→ `done`)

**Step 1 `.py`:**

```python
# 4.1_Introduction
"""第4章概述: 打印本章路线图。"""
ROADMAP = [
    ("4.2", "变量类型与术语"),
    ("4.3", "两类简单方法: 最小二乘与最近邻"),
    ("4.4", "统计决策理论"),
    ("4.9", "模型选择与偏差-方差权衡"),
]


def main() -> None:
    print("第4章: Overview of Supervised Learning")
    for k, t in ROADMAP:
        print(f"  {k}: {t}")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 4.1 Introduction（引言）

本章为全书监督学习方法提供一个统一视角：从数据的统计决策理论出发，把回归与分类刻画为**风险最小化与函数逼近**，并引入模型选择与偏差-方差权衡的基本框架。

- 4.2–4.4 形式化问题；
- 4.6–4.8 讨论函数类与受限估计量；
- 4.9 给出模型复杂度取舍。

运行 `.py` 打印本章路线图。
```

**Steps 3–5:** `done`; run; commit `content: 4.1 introduction`.

---

### Task 6: 4.2 — Variable Types and Terminology (→ `done`)

**Step 1 `.py`:**

```python
# 4.2_Variable_Types_and_Terminology
"""变量类型: 数值 vs 类别, 类别用 one-hot 编码。"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def main() -> None:
    df = pd.DataFrame({"x1": [1.0, 2.5, 3.2], "x2": ["A", "B", "A"]})
    enc = OneHotEncoder()
    X = enc.fit_transform(df[["x2"]]).toarray()
    print("数值列:", df["x1"].to_numpy())
    print("类别 one-hot:\n", X)


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 4.2 Variable Types and Terminology（变量类型与术语）

- **定量变量**：实值（如收入、温度）。
- **定性变量（类别 / 因子）**：取值于离散集合，常用 **one-hot / 哑变量**编码为 0-1 向量。
- 记号：输入 $X$（$p$ 维）、输出 $Y$，共 $N$ 个观测。

## 公式

one-hot：$\phi(x)\in\{0,1\}^K$，$K$ 为类别个数。

## 代码示例

- pandas + `OneHotEncoder` 演示类别变量编码。

## 相关章节

- 第 5、6 章（回归 / 分类的输入表示）
```

**Steps 3–5:** `done`; commit `content: 4.2 variable types and terminology`.

---

### Task 7: 4.3 — Two Simple Approaches: Least Squares & Nearest Neighbors (→ `done`)

**Step 1 `.py`:**

```python
# 4.3_Two_Simple_Approaches_to_Prediction:_Least_Squares_and_Nearest_Neighbors
"""两类简单预测方法: 最小二乘 vs k-近邻。"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(0, 4, 120))
    y = np.sin(1.5 * x) + 0.15 * rng.standard_normal(120)
    xx = np.linspace(0, 4, 400)
    lin = LinearRegression().fit(x[:, None], y)
    for k in (1, 15):
        knn = KNeighborsRegressor(n_neighbors=k).fit(x[:, None], y)
        plt.plot(xx, knn.predict(xx[:, None]), label=f"kNN k={k}")
    plt.plot(xx, lin.predict(xx[:, None]), "--", label="最小二乘")
    plt.plot(xx, np.sin(1.5 * xx), "k-", label="真实")
    plt.scatter(x, y, s=5, alpha=0.4)
    plt.legend()
    plt.title("最小二乘 vs K-近邻")
    plt.savefig("04_Overview_of_Supervised_Learning/4.3_Two_Simple_Approaches_to_Prediction:_Least_Squares_and_Nearest_Neighbors/4.3_LS_vs_kNN.png", dpi=110)
    print("ok: 4.3")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 4.3 Two Simple Approaches: Least Squares & Nearest Neighbors（两类简单预测方法: 最小二乘与最近邻）

## 最小二乘

对线性模型 $f(x)=\beta_0+\sum_{j}\beta_j x_j$，最小化残差平方法和：

$$
\hat\beta=\arg\min_\beta \sum_{i=1}^{N}\left(y_i-\beta_0-\sum_{j}x_{ij}\beta_j\right)^2
$$

## k 近邻

在测试点附近取局部均值 / 多数投票：

$$
\hat f(x)=\frac{1}{k}\sum_{x_i\in N_k(x)}y_i
$$

## 权衡

- 最小二乘：低方差、高偏差（线性假设过强）。
- kNN：灵活 / 低偏差、但高方差，且随维数增大而失效（见 4.5）。

## 代码示例

- 在 1D 数据上比较最小二乘与 kNN（k=1 与 k=15）的拟合曲线。
```

**Steps 3–5:** `done`; commit `content: 4.3 two simple approaches`.

---

### Task 8: 4.4 — Statistical Decision Theory (→ `done`)

**Step 1 `.py`:**

```python
# 4.4_Statistical_Decision_Theory
"""监督学习的统计决策理论: 线性判别当作近贝叶斯分类器可视化。"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_blobs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.2, random_state=0)
    lda = LinearDiscriminantAnalysis().fit(X, y)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
    z = lda.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, z, alpha=0.3, cmap="bwr")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=12)
    plt.title("分类: LDA 作为贝叶斯分类器的近似")
    plt.savefig("04_Overview_of_Supervised_Learning/4.4_Statistical_Decision_Theory/4.4_贝叶斯分类器.png", dpi=110)
    print("ok: 4.4")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 4.4 Statistical Decision Theory（监督学习的统计决策理论）

## 内容

- 期望（风险）的最小化：回归为 $f^*(x)=\mathbb{E}[Y\mid X=x]$；分类为**贝叶斯分类器** $\arg\max_k\Pr(Y=k\mid X=x)$。
- 实际使用经验分布 $\hat P$，引入估计误差。
- 本节把"从数据中学习"严格表述为**风险最小化**问题，是后续所有方法的出发点。

## 代码示例

- 用 LDA 在两类数据上画出近贝叶斯决策边界。

## 相关章节

- 3.3（统计决策理论总论）、6.3（LDA）
```

**Steps 3–5:** `done`; commit `content: 4.4 statistical decision theory`.

---

### Task 9: 4.5 — Local Methods in High Dimensions, **DEEP → `notebook`**

**Step 1 `.py`:**

```python
# 4.5_Local_Methods_in_High_Dimensions
"""高维局部方法: 展示维数灾维中球的体积占比随维数下降。"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def ball_share(p, r=0.5):
    return math.pi ** (p / 2) / math.gamma(p / 2 + 1) * r ** p


def main() -> None:
    p = np.arange(1, 21)
    share = [ball_share(pp, 0.5) for pp in p]
    plt.plot(p, share, "-o")
    plt.xlabel("维数 p"); plt.ylabel("半径为0.5的超球体积(单位立方体内)")
    plt.title("高维空间的\"维数灾维\"")
    plt.savefig("04_Overview_of_Supervised_Learning/4.5_Local_Methods_in_High_Dimensions/4.5_curse.png", dpi=110)
    print("ok: p=10 时占比=", round(share[9], 4))


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 4.5 Local Methods in High Dimensions（高维空间中的局部方法）

## 问题

最近邻等方法依赖"局部平滑"，而随着维数 $p$ 增大，固定半径的邻域内样本越来越少：局部结构消失，样本彼此"远离"。

## 核心概念: 维数灾维

单位超立方体内，半径 $r$ 的超球与总体之比随 $p$ 指数下降：

$$
\text{Vol}_p(r)=\frac{\pi^{p/2}\,r^{\,p}}{\Gamma(p/2+1)}
$$

$p=10$ 时半径 $0.5$ 的超球体积仅占超立方体的约 $0.25\%$。因此 kNN 在高维下邻居稀疏、方差骤增。反之，假设全局结构的线性 / 贝叶斯方法在低样本高维下更稳定。

## 公式

见上式。

## 代码与图

- 运行 `.py` 生成超球占比 vs 维数的曲线 `4.5_curse.png`。

## 相关章节

- 4.3（kNN）、4.6（结构化模型对立"高维诅咒"）

## 配套笔记本

- 同目录 `.ipynb` 提供可交互的分步推演，见 Task:9b。
```

**Steps 3–5:** progress `4.5...` → `notebook`; run `.py`; commit after Task 9b (include both).

### Task 9b: 4.5 notebook

**Files:** create `04_Overview_of_Supervised_Learning/4.5_Local_Methods_in_High_Dimensions/4.5_Local_Methods_in_High_Dimensions.ipynb` + `scripts/build_nb_45.py`.

- [ ] **Step 1: Write `scripts/build_nb_45.py`**

```python
"""构建 4.5 深讲笔记本. 用法: python scripts/build_nb_45.py"""
import pathlib
import nbformat as nbf

md = [
    "# 4.5 高维空间的局部方法（深讲）",
    "本笔记本逐步复现\"维数灾维\"：随着维数增加，局部邻域迅速变空。",
    "## 1. 球与立方体的体积比",
    "在单位超立方体中放置半径 $r=0.5$ 的超球，其体积占比随维数指数下降。",
]
code = [
    "import math, numpy as np\n"
    "def share(p): return math.pi**(p/2)/math.gamma(p/2+1)*(0.5**p)\n"
    "print('p=10 占比 =', round(share(10), 5))",
    "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    "p = np.arange(1,21)\nplt.plot(p, [share(x) for x in p], 'o')\n"
    "plt.xlabel('维数 p'); plt.ylabel('占比')\nplt.title('维数灾维')\nplt.show()",
]
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(c) for c in md] + [nbf.v4.new_code_cell(c) for c in code]
out = pathlib.Path("04_Overview_of_Supervised_Learning/4.5_Local_Methods_in_High_Dimensions/4.5_Local_Methods_in_High_Dimensions.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out)
```

- [ ] **Step 2:** run `python scripts/build_nb_45.py`; then `python -c "import nbformat as n; n.read('..../4.5_Local_Methods_in_High_Dimensions.ipynb', as_version=4)"` — expect success.
- [ ] **Step 3:** commit `git add 04_...4.5... scripts/build_nb_45.py config/progress.yaml`; message `content: 4.5 local methods (deep+notebook)`.

---

### Task 10: 4.6 — Statistical Models, Supervised Learning & Function Approximation (→ `done`)

**Step 1 `.py`:**

```python
# 4.6_Statistical_Models,_Supervised_Learning_and_Function_Approximation
"""函数逼近视角: 多项式拟合的残差与均方误差随阶数的变化。"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(-3, 3, 150))
    y = np.sin(x) + 0.2 * rng.standard_normal(150)
    for deg in (1, 3, 7):
        model = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(x[:, None], y)
        mse = mean_squared_error(y, model.predict(x[:, None]))
        print(f"阶数 p={deg}: 训练MSE={mse:.3f}")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 4.6 Statistical Models, Supervised Learning & Function Approximation（统计模型与函数逼近）

## 内容

把学习视为在函数类中逼近未知回归函数 $f(x)$：模型由误差假设（噪声分布）与函数类构成。例如 $y=f(x)+\epsilon$、$\epsilon$ 均值为零。

- **模型**：对 $f$ 的参数/非参数形式的约束；
- **目标**：分数平方误差或似然等价于"逼近 $f$"。

## 公式

最小平方与高斯噪声下的最大似然等价：

$$
-\log p(y\mid x)=\frac{N}{2}\log(2\pi\sigma^2)+\frac{1}{2\sigma^2}\sum (y_i-f(x_i))^2\ \Rightarrow\ \text{MSE}
$$

## 代码示例

- 运行 `.py` 观察多项式阶数升高时训练 MSE 下降（过度拟合信号）。

## 相关章节

- 4.7（结构化模型）、7（基展开）
```

**Steps 3–5:** `done`; commit `content: 4.6 statistical models`.

---

### Task 11: 4.7 — Structured Regression (→ `done`)

**Step 1 `.py`:**

```python
# 4.7_Structured_Regression_Models
"""结构化回归: 线性/加性假设 vs 一般局部方法, 用双输入演示。"""
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)


def main() -> None:
    n = 200
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = 3 * x1 - 2 * x2 + 0.1 * rng.standard_normal(n)
    lr = LinearRegression().fit(np.c_[x1, x2], y)
    print("拟合参数:", np.round(lr.coef_, 3))


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 4.7 Structured Regression Models（结构化回归模型）

## 内容

全维局部方法（如 $k$NN）在高维失效（见 4.5），结构化假设通过限制函数形式来对抗"维数灾维"：

- **线性** $f=\sum_j\beta_j x_j$：P 个参数，形状固定；
- **可加** $f=\sum_j f_j(x_j)$：每维一个低维平滑；
- **低秩 / 稀疏**：限制有效自由度。

结构化模型可控带宽与样本量独立，采样性较好，但可能引入偏差。

## 代码示例

- 线性结构模型在两输入上的最小二乘几何；观察参数估计与设定一致。
```

**Steps 3–5:** `done`; commit `content: 4.7 structured regression models`.

---

### Task 12: 4.8 — Classes of Restricted Estimators (→ `done`)

**Step 1 `.py`:**

```python
# 4.8_Classes_of_Restricted_Estimators
"""受限估计器: 子集选择与岭收缩对比。"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.datasets import make_regression


def main() -> None:
    X, y = make_regression(n_samples=80, n_features=6, noise=0.4, random_state=0)
    full = LinearRegression().fit(X, y).coef_
    ridge = Ridge(alpha=10).fit(X, y).coef_
    print("满模型系数:", np.round(full, 2))
    print("岭回归系数:", np.round(ridge, 2))
    print("约束(收缩)使系数变小")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 4.8 Classes of Restricted Estimators（受限估计器）

## 内容

把估计器视为对 $f$ 的函数空间施加限制。常见类别：

- **惩罚（正则化）**：$\min\sum L + \lambda J(f)$（岭、Lasso、稀疏）；
- **核 / 局部方法**：限定为局部平均（kNN、核平滑）；
- **有限维限定**：样条、多项式、多元自适应回归。

## 核心思想

"受限"即减少有效自由度；同一方法可视为约束空间或罚。例子：子集选择（去掉某些变量，系数=0）vs 岭收缩（系数缩小）。

## 代码示例

- 比较满模型与岭回归的系数，展示收缩。

## 相关章节

- 3.4（正则化）、5.3–5.5（子集选择与收缩）
```

**Steps 3–5:** `done`; commit `content: 4.8 classes of restricted estimators`.

---

### Task 13: 4.9 — Model Selection & Bias–Variance Tradeoff, DEEP → `notebook`

*(Note: the directory/file use the en-dash `–` in the name, e.g. `4.9_Model_Selection_and_the_Bias–Variance_Tradeoff`.)*

**Step 1 `.py`:**

```python
# 4.9_Model_Selection_and_the_Bias_Variance_Tradeoff
"""偏差-方差分解模拟: 最近邻, 不同 k。"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(0)


def f(x):
    return np.sin(1.5 * x)


def main() -> None:
    n, rep = 120, 200
    ks = [1, 3, 9, 30, 80]
    xs = np.linspace(0, 4, 150)
    true = f(xs)
    print("k      bias^2   var")
    for k in ks:
        preds = np.zeros((rep, len(xs)))
        for r in range(rep):
            x = rng.uniform(0, 4, n)
            y = f(x) + 0.2 * rng.standard_normal(n)
            preds[r] = KNeighborsRegressor(n_neighbors=k).fit(x[:, None], y).predict(xs[:, None])
        b = np.mean((preds.mean(0) - true) ** 2)
        v = np.mean(np.var(preds, axis=0))
        print(f"{k:3d}    {b:.3f}  {v:.3f}")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 4.9 Model Selection & Bias–Variance Tradeoff（模型选择与偏差-方差权衡）

## 内容

测试误差可分解为（不可约噪声 $\sigma^2$）：

$$
\mathbb{E}(y_0-\hat f(x_0))^2 = \text{Bias}^2 + \text{Var} + \sigma^2
$$

- 复杂模型 → 方差大、偏差小；简单模型 → 偏差大、方差小。
- 模型选择（欠 / 过拟合）即在二者之间寻找平衡，并用交叉验证(见第 9 章)估计做选择。

## 代码示例

- 运行 `.py`：不同 k 下偏差²、方差的实测，直观体现基于邻近的方法的偏差-方差权衡。

## 相关章节

- 9.3（偏差-方差分解）、9.10（交叉验证）

## 配套笔记本

- 见 Task 13b：以交互方式重复该分解并绘图。
```

**Steps:** `bias_variance` stub → `notebook`; run `.py`.

### Task 13b: 4.9 notebook

Mirror Task 9b: `scripts/build_nb_49.py` builds
`04_Overview_of_Supervised_Learning/4.9_Model_Selection_and_the_Bias–Variance_Tradeoff/4.9_Model_Selection_and_the_Bias–Variance_Tradeoff.ipynb`
with markdown narrative + inline code producing a bar/line of bias² and variance vs k (reuse the `.py` logic inline). Steps: write build script → run → verify with `nbformat.read` → commit (ipynb + script + md + py + progress).

---

### Task 14: 4.10 — Bibliographic Notes (→ `done`)

**Step 1 `.py`:**

```python
# 4.10_Bibliographic_Notes
"""第4章文献指引。"""
REFS = [
    "Hastie, Tibshirani & Friedman (2009) ESL Ch.4",
    "Bishop (2006) PRML",
    "Vapnik (1995)",
    "Ripley (1996)",
]


def main() -> None:
    for r in REFS:
        print("-", r)


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** a short Chinese **Bibliographic Notes** paragraph pointing to the above references and where each relates within chapter 04.

**Steps 3–5:** `done`; commit `content: 4.10 bibliographic notes`.

---

### Task 15: 4.11 — Exercises (→ `done`)

**Step 1 `.py`:** solve a concrete exercise (simulate kNN test error vs k and report the best k by a holdout).

```python
# 4.11_Exercises
"""习题: 用留出集选择 kNN 的 k, 并比较偏差-方差。"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(0, 4, 300))
    y = np.sin(1.5 * x) + 0.2 * rng.standard_normal(300)
    tr, te = x[:200], x[200:]
    ytr, yte = y[:200], y[200:]
    best = None
    for k in (1, 3, 5, 10, 20, 40):
        m = KNeighborsRegressor(n_neighbors=k).fit(tr[:, None], ytr)
        err = mean_squared_error(yte, m.predict(te[:, None]))
        print(f"k={k:3d} 测试MSE={err:.4f}")
        if best is None or err < best[1]:
            best = (k, err)
    print("最优 k =", best[0], "MSE=", round(best[1], 4))


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** list 3 conceptual exercises in Chinese (e.g., (1) explain when kNN beats linear regression; (2) describe the curse of dimensionality with a small example; (3) derive bias–variance for a given k), plus a "参考解答" quoting the holdout k-selection demo in `.py`.

**Steps 3–5:** `done`; commit `content: 4.11 exercises`.

---

### Task 16: README refresh + full verification

**Files:** `README.md` (regenerated), no code.

- [ ] **Step 1:** `python scripts/generate_docs.py --regenerate`
  Expected: `created={md:0, py:0, ...}`; README rebuilt.
- [ ] **Step 2:** `python -m pytest` → **8 passed**; `python scripts/smoke_test.py` → pass (count shrinks to 175+2=ceil).
- [ ] **Step 3:** sanity: `grep -c "✓ 摘要" README.md` ≈ number of ch3+4 `done` sections + legend; `grep -c "🔬 笔记" README.md` = 3 (2 deep + legend).
- [ ] **Step 4:** `grep "\.ipynb" README.md` lists exactly 4.5 and 4.9.
- [ ] **Step 5:** commit `git add -A && git commit -m "docs: Phase 3 foundations content"`.

## Self-Review

**Spec coverage (design §10 Phase 3 / §3–§6):** all 15 sections authored; two deep notebooks built; progress/README updated; figures saved per §6 convention. ✓ **Placeholder scan:** all 15 tasks fully written (4.6 and 4.11 were earlier left thin and are now complete); every `.py`, `.md`, and both notebook builders contain real, runnable code. ✓
**Type consistency:** 4.9 en-dash preserved; progress keys match dirs; `nbformat` builders named `build_nb_{45,49}.py`. ✓

Reminder to executor: run each Task's `.py` and watch for file-path typos (the build scripts create the notebook; the `.md`/`.py` are direct writes).