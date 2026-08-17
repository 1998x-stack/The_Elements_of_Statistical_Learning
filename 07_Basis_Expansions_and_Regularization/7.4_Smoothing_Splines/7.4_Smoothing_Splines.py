# 7.4_Smoothing_Splines
"""平滑样条: 光滑度参数 vs 偏差-方差。"""
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
    plt.legend()
    plt.title("平滑样条: 光滑度 vs 偏差")
    plt.savefig("07_Basis_Expansions_and_Regularization/7.4_Smoothing_Splines/7.4_smoothing.png", dpi=110)
    print("ok: 7.4")


if __name__ == "__main__":
    main()