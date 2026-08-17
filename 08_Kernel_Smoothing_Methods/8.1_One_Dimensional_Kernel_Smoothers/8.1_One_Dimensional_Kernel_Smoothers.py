# 8.1_One_Dimensional_Kernel_Smoothers
"""一维核平滑: 高斯核权重对噪声曲线的估计。"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def gauss_smooth(x, y, xx, h=0.15):
    out = np.empty_like(xx)
    for j, t in enumerate(xx):
        w = np.exp(-((x - t) / h) ** 2)
        out[j] = np.sum(w * y) / np.sum(w)
    return out


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 150))
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(150)
    xx = np.linspace(0, 1, 300)
    plt.scatter(x, y, s=6, alpha=0.4)
    for h in (0.05, 0.15, 0.4):
        plt.plot(xx, gauss_smooth(x, y, xx, h), label=f"核宽 h={h}")
    plt.plot(xx, np.sin(6 * xx), "k--", label="真实")
    plt.legend()
    plt.title("一维核平滑: 带宽的影响")
    plt.savefig("08_Kernel_Smoothing_Methods/8.1_One_Dimensional_Kernel_Smoothers/8.1_kernel.png", dpi=110)
    print("ok: 8.1")


if __name__ == "__main__":
    main()