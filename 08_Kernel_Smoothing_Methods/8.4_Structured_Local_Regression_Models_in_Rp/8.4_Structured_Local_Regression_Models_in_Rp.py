# 8.4_Structured_Local_Regression_Models_in_Rp
"""结构化局部回归: 各维分别加性平滑(可加)在 2D 数据上。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def smooth1d(x, y, xx, h=0.2):
    out = np.empty_like(xx)
    for j, t in enumerate(xx):
        w = np.exp(-((x - t) / h) ** 2)
        out[j] = np.sum(w * y) / np.sum(w)
    return out


def main() -> None:
    n = 200
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(4 * x1) + np.cos(4 * x2) + 0.2 * rng.standard_normal(n)
    # 可加近似: 分别对 x1、x2 平滑
    xx = np.linspace(0, 1, 200)
    plt.plot(xx, np.sin(4 * xx), "k--", label="真实 f1")
    plt.plot(xx, smooth1d(np.sort(x1), y[np.argsort(x1)], xx), label="平滑 f1(x1)")
    plt.legend()
    plt.title("结构化(可加)局部平滑: 维度一")
    plt.savefig("08_Kernel_Smoothing_Methods/8.4_Structured_Local_Regression_Models_in_Rp/8.4_structured.png", dpi=110)
    print("ok: 8.4")


if __name__ == "__main__":
    main()