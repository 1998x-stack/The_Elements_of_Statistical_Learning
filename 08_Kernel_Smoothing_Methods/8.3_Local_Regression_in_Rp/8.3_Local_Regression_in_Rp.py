# 8.3_Local_Regression_in_Rp
"""Rp 中的局部回归: 局部常数(核平均) vs 局部线性。"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def local_linear(x, y, xx, h=0.25):
    out = np.empty_like(xx)
    for j, t in enumerate(xx):
        w = np.exp(-((x - t) / h) ** 2)
        W = np.diag(w)
        Xd = np.c_[np.ones_like(x), x - t]
        beta = np.linalg.solve(Xd.T @ W @ Xd, Xd.T @ W @ y)
        out[j] = beta[0]
    return out


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 150))
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(150)
    xx = np.linspace(0, 1, 300)
    plt.scatter(x, y, s=6, alpha=0.4)
    plt.plot(xx, local_linear(x, y, xx, 0.25), label="局部线性 h=0.25")
    plt.plot(xx, np.sin(6 * xx), "k--", label="真实")
    plt.legend()
    plt.title("局部线性回归")
    plt.savefig("08_Kernel_Smoothing_Methods/8.3_Local_Regression_in_Rp/8.3_locallinear.png", dpi=110)
    print("ok: 8.3")


if __name__ == "__main__":
    main()