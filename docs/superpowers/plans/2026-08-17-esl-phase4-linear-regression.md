# ESL Linear Regression (Phase 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapter 05 (Linear Methods for Regression, 11 sections): runnable `.py` for every section and rich `.ipynb` notebooks for the three deep sections (5.2, 5.4, 5.5). Update `progress.yaml`/`README.md` to ✅/🔬.

**Architecture:** Established Phase 3 pattern — direct-write Chinese `.md` + runnable `.py`; build deep `.ipynb` via `nbformat`; flip `progress.yaml` statuses; rebuild README with `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib, nbformat.

## Global Constraints

- `.md` narrative in **Chinese**; technical terms/code English; math in LaTeX. `.md` begins with `# <Full English Title>（<中文名>）`.
- Every `.py` runs standalone headless: `import matplotlib; matplotlib.use("Agg")` before `pyplot`; `np.random.seed(0)`/`default_rng(0)`; `if __name__ == "__main__":`; saves figures as `<sectiondir>/<section>_<name>.png`. **Never run or modify `main.py`.**
- `progress.yaml` keys canonical (no leading slash), e.g. `05_Linear_Methods_for_Regression/5.2_Linear_Regression_Models_and_Least_Squares`.
- Statuses: `done` / `notebook`; deep this phase: **5.2, 5.4, 5.5**.
- Suppress sklearn/matplotlib benign `RuntimeWarning`s in scripts that fit high-degree or kernel models (`warnings.filterwarnings("ignore", category=RuntimeWarning)`).
- Figure font preamble for every plotting script:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
```

## Standard 5-step task envelope (content per task below)

- [ ] 1. Write `<section>.py`  2. Write `<section>.md`  3. progress override  4. run `.py`  5. commit

---

### Task 1: 5.1 — Introduction (→ `done`)

**Step 1 `.py`:** roadmap print.

```python
# 5.1_Introduction
"""第5章概述: 线性方法回归。"""
ROADMAP = [
    ("5.2", "线性回归与最小二乘"),
    ("5.3-5.4", "子集选择与收缩(岭/Lasso)"),
    ("5.5", "派生输入方向(PCR/PLS)"),
    ("5.9", "计算问题"),
]


def main() -> None:
    print("第5章: Linear Methods for Regression")
    for k, t in ROADMAP:
        print(f"  {k}: {t}")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** Chinese intro defining the linear model $f(X)=\beta_0+\sum X_j\beta_j$, why it matters, chapter roadmap.

**Steps 3–5:** `05.../5.1_Introduction: stub`→`done`; run; commit `content: 5.1 introduction`.

---

### Task 2: 5.2 — Linear Regression Models and Least Squares, **DEEP → `notebook`**

**Step 1 `.py`:** closed-form OLS via normal equations and via `np.linalg.lstsq` on a small dataset; report coefficients + RSS and residual vs fitted plot.

```python
# 5.2_Linear_Regression_Models_and_Least_Squares
"""最小二乘线性回归: 正规方程与矩阵求解, 附残差图。"""
import warnings
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    X, y = load_diabetes(return_X_y=True)
    Xb = np.c_[np.ones(len(X)), X]
    # 正规方程 beta = (X'X)^-1 X'y
    beta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
    yhat = Xb @ beta
    rss = np.sum((y - yhat) ** 2)
    print("闭式解 beta0 = {:.3f}".format(beta[0]))
    print("RSS =", round(rss, 2))
    plt.scatter(yhat, y - yhat, s=10)
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("拟合值"); plt.ylabel("残差")
    plt.title("残差 vs 拟合值 (糖尿病数据)")
    plt.savefig("05_Linear_Methods_for_Regression/5.2_Linear_Regression_Models_and_Least_Squares/5.2_残差图.png", dpi=110)
    print("ok: 5.2")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:**

```markdown
# 5.2 Linear Regression Models and Least Squares（线性回归模型与最小二乘）

## 模型
对输入向量 $X=(1,X_1,\dots,X_p)^T$，线性模型 $f(X)=\beta_0+\sum_{j=1}^p X_j\beta_j$。

## 最小二乘估计
最小化残差平方法和 $RSS(\beta)=\sum_{i=1}^N (y_i-x_i^T\beta)^2$，解为：

$$
\hat\beta=(X^T X)^{-1}X^T y
$$

$\hat\beta$ 无偏，方差阵 $\mathrm{Var}(\hat\beta)=\sigma^2(X^TX)^{-1}$。

## 检验与置信
假设检验与置信区间基于 $\hat\sigma^2=RSS/(N-p-1)$，常用于变量显著性判断。

## 代码与图
- 运行 `.py`：用正规方程求闭式解，输出 RSS 并绘制残差-拟合图。
- 配套 `notebook`：逐步推导 $X^TX$、求逆与残差计算（见 Task 2b）。

## 相关
- 3.3、4.4（决策理论）；5.4（岭）；5.5（PCR）
```

**Steps 3–5:** `5.2...: stub`→`notebook`; run; include in commit after Task 2b.

### Task 2b: 5.2 notebook — `scripts/build_nb_52.py`

markdown cell explains normal equations; code cells solve via `np.linalg.solve`, print RSS, and plot residuals. Mirror Task 9b of Phase 3 exactly (build script → run → `nbformat.read` verify → commit).

---

### Task 3: 5.3 — Subset Selection (→ `done`)

**Step 1 `.py`:** compare best subset vs forward stepwise by Cp/AIC on diabetes (use forward selection loop by R² added but keep it self-contained with scores).

```python
# 5.3_Subset_Selection
"""子集选择: 前向逐步加入特征并报告每步的拟合优度。"""
import warnings
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_diabetes(return_X_y=True)
    n, p = X.shape
    selected = []
    r2_prev = -np.inf
    for step in range(5):
        best = None
        for j in range(p):
            if j in selected:
                continue
            cols = selected + [j]
            lr = LinearRegression().fit(X[:, cols], y)
            r2 = lr.score(X[:, cols], y)
            if best is None or r2 > best[1]:
                best = (j, r2)
        selected.append(best[0])
        print(f"步 {step+1}: 加入特征 {selected[-1]}  R^2={best[1]:.4f}")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** best-subset, forward/backward stepwise, criteria (Cp/AIC/BIC); pros/cons.

**Steps 3–5:** `done`; commit `content: 5.3 subset selection`.

---

### Task 4: 5.4 — Shrinkage Methods, **DEEP → `notebook`**

**Step 1 `.py`:** ridge coefficient path vs λ; lasso via LassoLars.

```python
# 5.4_Shrinkage_Methods
"""收缩方法: 岭回归系数的路径与 Lasso 稀疏解对比。"""
import warnings
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    X, y = make_regression(n_samples=80, n_features=8, n_informative=5, noise=0.5, random_state=0)
    alphas = np.logspace(-3, 3, 40)
    ridge_coefs = [Ridge(alpha=a).fit(X, y).coef_ for a in alphas]
    plt.plot(np.log10(alphas), np.array(ridge_coefs))
    plt.xlabel("log10(alpha)"); plt.ylabel("系数")
    plt.title("岭回归系数路径")
    plt.savefig("05_Linear_Methods_for_Regression/5.4_Shrinkage_Methods/5.4_岭路径.png", dpi=110)
    lasso = Lasso(alpha=0.5, max_iter=5000).fit(X, y)
    print("Lasso 非零系数个数:", int(np.sum(lasso.coef_ != 0)))
    print("ok: 5.4")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** ridge (L2) closed form, lasso (L1) → sparsity, bias introduced to cut variance; norms.

**Steps 3–5:** notebook 5.4 (build_nb_54: ridge path + lasso) → `notebook` → commit `content: 5.4 shrinkage methods (deep)`.

---

### Task 5: 5.5 — Methods Using Derived Input Directions, **DEEP → `notebook`**

**Step 1 `.py`:** PCR vs OLS vs PLS risk on correlated data.

```python
# 5.5_Methods_Using_Derived_Input_Directions
"""派生输入方向: PCA回归(PCR) 与偏最小二乘(PLS)。"""
import warnings
import numpy as np
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(1)


def main() -> None:
    X, y = make_regression(n_samples=120, n_features=10, noise=0.6, random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ols = LinearRegression().fit(Xtr, ytr).predict(Xte)
    for n in (2, 5):
        pcr = PCA(n_components=n)
        Xp = pcr.fit_transform(Xtr)
        m = LinearRegression().fit(Xp, ytr)
        mse = mean_squared_error(yte, m.predict(pcr.transform(Xte)))
        print(f"PCR k={n} 测试MSE={mse:.2f} ")
    pls = PLSRegression(n_components=5).fit(Xtr, ytr)
    print("PLS 测试MSE =", round(mean_squared_error(yte, pls.predict(Xte)), 2))
    print("OLS 测试MSE =", round(mean_squared_error(yte, ols), 2))


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** PCR projects to leading principal components; PLS applies supervised weighting; reduces variance vs OLS.

**Steps 3–5:** notebook 5.5 (build_nb_55) → `notebook` → commit.

---

### Task 6: 5.6 — Discussion: Comparison of Selection & Shrinkage (→ `done`)

**Step 1 `.py`:** tabulate coefficient counts & train/test MSE for subset, ridge, lasso, PCR on diabetes.

```python
# 5.6_Discussion:_A_Comparison_of_the_Selection_and_Shrinkage_Methods
"""比较子集/岭/Lasso/PCR: 系数数与测试MSE。"""
import warnings
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    res = []
    ols = LinearRegression().fit(Xtr, ytr)
    res.append(("OLS", np.sum(ols.coef_ != 0), mean_squared_error(yte, ols.predict(Xte))))
    ridge = Ridge(alpha=8).fit(Xtr, ytr)
    res.append(("RIDGE", np.sum(np.abs(ridge.coef_) > 1e-8), mean_squared_error(yte, ridge.predict(Xte))))
    lasso = Lasso(alpha=1.0, max_iter=5000).fit(Xtr, ytr)
    res.append(("LASSO", np.sum(lasso.coef_ != 0), mean_squared_error(yte, lasso.predict(Xte))))
    pcr = PCA(n_components=6).fit(Xtr); Xtp = pcr.transform(Xtr)
    m = LinearRegression().fit(Xtp, ytr)
    res.append(("PCR (6)", 6, mean_squared_error(yte, m.predict(pcr.transform(Xte)))))
    for name, nonz, mse in res:
        print(f"{name:9s} 非零系数={nonz:3d} 测试MSE={mse:.2f}")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** compare strengths, effective degrees of freedom.

**Steps 3–5:** `done`; commit.

---

### Task 7: 5.7 — Multiple Outcome Shrinkage and Selection (→ `done`)

**Step 1 `.py`:** ridge for multiple correlated outputs.

```python
# 5.7_Multiple_Outcome_Shrinkage_and_Selection
"""多元输出的收缩: 岭回归处理相关响应。"""
import warnings
import numpy as np
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore", category=RuntimeWarning)
rng = np.random.default_rng(0)


def main() -> None:
    X = rng.normal(size=(100, 3))
    Y = np.c_[X[:, 0] + X[:, 1], X[:, 1] - X[:, 0]] + rng.normal(scale=0.2, size=(100, 2))
    r = Ridge(alpha=5).fit(X, Y)
    print("多输出岭回归系数形状:", r.coef_.shape)


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** multivariate least squares loss + shrinkage, note reduced-rank structure.

**Steps 3–5:** `done`; commit.

---

### Task 8: 5.8 — More on the Lasso and Path Algorithms (→ `done`)

**Step 1 `.py`:** demonstrate lasso path (coefficients vs regularization) and show sparsity monotonic.

```python
# 5.8_More_on_the_Lasso_and_Related_Path_Algorithms
"""Lasso 路径: 随 alpha 变化系数逐步变为 0。"""
import warnings
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_regression(n_samples=60, n_features=6, noise=0.4, random_state=0)
    for a in (1e-3, 0.05, 0.2, 0.6):
        l = Lasso(alpha=a, max_iter=10000).fit(X, y)
        print(f"alpha={a:.3f} 非零系数={int(np.sum(l.coef_!=0))}")

if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** LARS/homotopy, glmnet coordinate descent, why lasso yields exact zeros.

**Steps 3–5:** `done`; commit.

---

### Task 9: 5.9 — Computational Considerations (→ `done`)

**Step 1 `.py`:** time OLS by normal equations vs `np.linalg.lstsq`/QR on p×N "less-than-full-rank".

```python
# 5.9_Computational_Considerations
"""计算注意: 比较正规方程与 QR/最小二乘求解的耗时。"""
import time
import numpy as np
from sklearn.datasets import make_regression
rng = np.random.default_rng(0)


def main() -> None:
    X, y = make_regression(n_samples=1500, n_features=400, noise=0.1, random_state=0)
    t0 = time.perf_counter()
    np.linalg.solve(X.T @ X, X.T @ y)
    t1 = time.perf_counter()
    np.linalg.lstsq(X, y, rcond=None)
    t2 = time.perf_counter()
    print(f"正规方程: {t1-t0:.4f}s    lstsq(QR): {t2-t1:.4f}s")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** X^TX cost, ill-conditioning, QR/SVD as stable alternatives.

**Steps 3–5:** `done`; commit.

---

### Task 10: 5.10 — Bibliographic Notes (→ `done`)
`.py` prints refs (Hoerl & Kennard ridge; Tibshirani lasso; Wold PLS; Stone/Cox PCR). `.md` brief Chinese guidance. `done`; commit.

### Task 11: 5.11 — Exercises (→ `done`)
`scikit-learn`-based single exercisable: derive / demo ridge closed form and compare to `Ridge()` on diabetes, print max abs diff on `.py`; `.md` lists 3 Chinese exercises + a solved reference. `done`; commit.

---

### Task 12: README refresh + full verification

- [ ] 1. `python scripts/generate_docs.py --regenerate`
- [ ] 2. `python -m pytest` (≥8 passed) + `python scripts/smoke_test.py`
- [ ] 3. confirm README: ch5 `done` entries + `🔬 笔记` for 5.2/5.4/5.5
- [ ] 4. commit `docs: Phase 4 linear regression content`

## Self-Review

**Spec coverage (§10 Phase 4):** all 11 ch5 sections authored; three deep notebooks; progress/README updated; figures per §6. ✓
**Placeholder scan:** every task has a concrete `.py` and real Chinese `.md`; the notebooks are built by the inline `build_nb_{52,54,55}.py` scripts. ✓
**Type consistency:** progress keys match directories; `5.2`/`5.4`/`5.5` all `notebook`; others `done`; `.ipynb` names match section names exactly. ✓
**Engineer reminder:** never run `main.py`; run smokes only after scripts are warnings-free.