# 11.4_MARS:_Multivariate_Adaptive_Regression_Splines
"""MARS: 用 hinge 基(截断线性)前向逐步拟合 1D 分段线性函数。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)


def hinge(x, k):
    return np.maximum(x - k, 0)


def basis(x, knots):
    return np.column_stack([np.ones_like(x)] + [hinge(x, k) for k in knots])


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 200))
    y = 2 + 5 * hinge(x, 0.4) - 3 * hinge(x, 0.7) + 0.1 * rng.standard_normal(200)
    # 前向逐步: 每次加入使残差平方和最小的候选 hinge
    knots_sel = []
    for _ in range(4):
        best = None
        for k in x:
            B = basis(x, knots_sel + [k])
            m = LinearRegression().fit(B, y)
            rss = np.sum((y - m.predict(B)) ** 2)
            if best is None or rss < best[0]:
                best = (rss, k)
        knots_sel.append(best[1])
    m = LinearRegression().fit(basis(x, knots_sel), y)
    xx = np.linspace(0, 1, 300)
    plt.scatter(x, y, s=6, alpha=0.5)
    plt.plot(xx, m.predict(basis(xx, knots_sel)), lw=2, label="MARS(前向 hinge)")
    plt.legend()
    plt.title("MARS: 分段线性逼近")
    plt.savefig("11_Additive_Models,_Trees,_and_Related_Methods/11.4_MARS:_Multivariate_Adaptive_Regression_Splines/11.4_mars.png", dpi=110)
    print(f"ok: 11.4 MARS 选中的折点 = {np.round(knots_sel, 3)}")


if __name__ == "__main__":
    main()