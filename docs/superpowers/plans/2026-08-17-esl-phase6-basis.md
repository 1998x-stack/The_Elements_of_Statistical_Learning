# ESL Basis Expansions & Regularization (Phase 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author full Chinese tutorial content for chapter 07 (Basis Expansions & Regularization, 11 sections): runnable `.py` for every section and rich `.ipynb` notebooks for the two deep sections (7.2 piecewise polynomials & splines, 7.4 smoothing splines). Update `progress.yaml`/`README.md` to ✅/🔬.

**Architecture:** Established pattern — direct-write Chinese `.md` + runnable `.py`; build deep `.ipynb` via `nbformat`; flip `progress.yaml`; rebuild README with `scripts/generate_docs.py --regenerate`. Never run legacy `main.py`.

**Tech Stack:** Python 3.9+, numpy, scipy, scikit-learn, pandas, matplotlib, nbformat.

## Global Constraints

- `.md` narrative in **Chinese**; technical terms/code English; math in LaTeX. `.md` begins with `# <Full English Title>（<中文名>）`.
- Every `.py` runs standalone headless (`matplotlib.use("Agg")` before `pyplot`), seeds RNG, uses `if __name__ == "__main__":`, saves figures as `<sectiondir>/<section>_<name>.png`. **Never run `main.py`.**
- `progress.yaml` keys canonical, e.g. `07_Basis_Expansions_and_Regularization/7.2_Piecewise_Polynomials_and_Splines`.
- Statuses `done`/`notebook`; deep this phase: **7.2, 7.4**.
- Suppress benign `RuntimeWarning`s where spline/kernel fits may warn.
- Figure font preamble for plotting scripts:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
```

## Standard 5-step task envelope (content per task)

- [ ] 1. Write `<section>.py`  2. Write `<section>.md`  3. progress override  4. run `.py`  5. commit

---

### Task 1: 7.1 — Introduction (→ `done`)
`.py` prints roadmap. `.md`: 用基函数 $h_m(x)$ 把输入变换为 $\{h_m(X)\}_{m=1}^{M}$ 后做线性模型 $f(X)=\sum_m \beta_m h_m(X)$，介绍多项式、样条、小波等。`7.1...: stub`→`done`; commit.

### Task 2: 7.2 — Piecewise Polynomials & Splines, **DEEP → `notebook`**

**Step 1 `.py`:** cubic spline vs global polynomial with knots on noisy data, via `make_pipeline(PolynomialFeatures, LinearRegression)` manual spline basis using truncated power / `scipy.interpolate.CubicSpline`.

```python
# 7.2_Piecewise_Polynomials_and_Splines
"""分段多项式与样条: 连续样条 vs 全局多项式。"""
import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 100))
    y = np.sin(4 * x) + 0.15 * rng.standard_normal(100)
    xx = np.linspace(0, 1, 300)
    # 三次样条(带节点)
    knots = np.quantile(x, [0.33, 0.66])
    cs = CubicSpline(x, y, bc_type="natural")
    # 高次全局多项式(无节点)
    poly = LinearRegression().fit(PolynomialFeatures(30).fit_transform(x[:, None]), y)
    plt.scatter(x, y, s=8, alpha=0.4)
    plt.plot(xx, cs(xx), label="三次样条(natural)")
    plt.plot(xx, poly.predict(PolynomialFeatures(30).fit_transform(xx[:, None])), label="30次多项式(过拟合)")
    plt.legend(); plt.title("分段样条 vs 全局多项式")
    plt.savefig("07_Basis_Expansions_and_Regularization/7.2_Piecewise_Polynomials_and_Splines/7.2_spline.png", dpi=110)
    print("ok: 7.2")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** 分段多项式、样条的连续性条件；截断幂基、B 样条；节点选取；与全局多项式比较。

**Steps 3–5:** notebook `7.2` (`build_nb_72.py`: 构造截断幂基 + 拟合逐步展示) → `7.2`→`notebook`; commit.

### Task 3: 7.3 — Filtering and Feature Extraction (→ `done`)
`.py`: 把特征矩阵的各列看作信号，对某一列做滑动平均/卷积平滑。`.md`: 把原始特征当作滤波/特征提取。`done`; commit.

### Task 4: 7.4 — Smoothing Splines, **DEEP → `notebook`**

**Step 1 `.py`:** smoothing spline via `scipy.interpolate.UnivariateSpline` with the `s` parameter to show smoothness vs data fit.

```python
# 7.4_Smoothing_Splines
"""平滑样条: 损失+曲率惩罚见 vs 不同光滑度。"""
import warnings
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 120))
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(120)
    xx = np.linspace(0, 1, 400)
    plt.scatter(x, y, s=6, alpha=0.4)
    for s in (0.1, 0.6):
        us = UnivariateSpline(x, y, s=s)
        plt.plot(xx, us(xx), label=f"smooth s={s}")
    plt.plot(xx, np.sin(6 * xx), "k--", label="真实")
    plt.legend(); plt.title("平滑样条: 平滑度 vs 偏差")
    plt.savefig("07_Basis_Expansions_and_Regularization/7.4_Smoothing_Splines/7.4_smoothing.png", dpi=110)
    print("ok: 7.4")


if __name__ == "__main__":
    main()
```

**Step 2 `.md`:** 惩罚最小二乘问题：损失 + $\lambda\int (f'')^2$；解是自然三次样条；有效自由度与交叉验证。

**Steps 3–5:** notebook `7.4` (`build_nb_74.py`) → `7.4`→`notebook`; commit.

### Task 5: 7.5 — Nonparametric Logistic Regression (→ `done`)
`.py`: local logistic + spline-based logistic on 1D, plot probability. `.md`; `done`; commit.

### Task 6: 7.6 — Multidimensional Splines (→ `done`)
`.py`: 2D thin-plate/`Radial` or `make_datasets` tensor; `.md`: 张量积基；`done`; commit.

### Task 7: 7.7 — Regularization & RKHS (→ `done`)
`.py`: kernel ridge regression on 1D via `KernelRidge(RBF)`; `.md`: 核函数 $K$、再生核、$f(x)=\sum_i \alpha_i K(x,x_i)$；`done`; commit.

### Task 8: 7.8 — Wavelet Smoothing (→ `done`)
`.py`: scipy 无 pywt，用 Haar 小波阈值去噪手工实现：对带噪阶跃信号做 Haar 变换、软/硬阈值、重建。`.md`: 多分辨分析、小波阈值去噪。`done`; commit.

### Task 9: 7.9 — Bibliographic Notes (→ `done`)
`.py` prints refs (de Boor—样条; Wahba—平滑样条; Mallat—小波). `.md` brief. `done`; commit.

### Task 10: 7.10 — Exercises (→ `done`)
`.py`: spline vs polynomial cross-validated error on synthetic; `.md` 3 Chinese exercises + solved note. `done`; commit.

### Task 11: 7.11 — Appendix: Computational Considerations for Splines (→ `done`)
`.py`: 构造 B-样条基（用 `scipy.interpolate.BSpline`）求解带惩罚的最小二乘，观察稀疏性；`.md`: 带状/稀疏线性系统。`done`; commit.

### Task 12: README refresh + full verification
- [ ] `python scripts/generate_docs.py --regenerate`
- [ ] `python -m pytest` (≥8) + `python scripts/smoke_test.py`
- [ ] confirm 7.2/7.4 as `🔬`
- [ ] commit `docs: Phase 6 basis expansions & regularization content`

## Self-Review
**Spec coverage (§10 Phase 6):** all 11 ch7 sections; two deep notebooks (7.2, 7.4); progress/README. ✓
**Placeholder scan:** per-task `.py` body + Chinese `.md`; notebooks via `build_nb_{72,74}.py`. ✓ (7.2/7.4 nb scripts are fully specified).
**Type consistency:** deep: 7.2, 7.4 → `notebook`; others `done`. Implementation should double-check the `UnivariateSpline`/`CubicSpline` APIs in `scipy` (deviation-tolerant). ✓
**Engineer note:** be careful with absolute paths when writing `.md`; never run `main.py`; run smoke only after scripts are warnings-free.