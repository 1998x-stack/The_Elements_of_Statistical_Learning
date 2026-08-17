# 8.7_Radial_Basis_Functions_and_Kernels
"""径向基函数: RBF 网络(中心扩展)逼近曲线。"""
import warnings
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 120))
    y = np.sin(6 * x) + 0.15 * rng.standard_normal(120)
    xx = np.linspace(0, 1, 300)
    rbf = RBFInterpolator(x[:, None], y, smoothing=0.05, kernel="thin_plate_spline")
    plt.scatter(x, y, s=8, alpha=0.5)
    plt.plot(xx, rbf(xx[:, None]), label="RBF (thin-plate)")
    plt.legend()
    plt.title("径向基函数展开")
    plt.savefig("08_Kernel_Smoothing_Methods/8.7_Radial_Basis_Functions_and_Kernels/8.7_rbf.png", dpi=110)
    print("ok: 8.7")


if __name__ == "__main__":
    main()