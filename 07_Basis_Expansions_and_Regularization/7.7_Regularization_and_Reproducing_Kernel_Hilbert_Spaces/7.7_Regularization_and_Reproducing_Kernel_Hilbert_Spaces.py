# 7.7_Regularization_and_Reproducing_Kernel_Hilbert_Spaces
"""正则化与再生核: 核岭回归(RBF核)在噪声曲线上的拟合。"""
import warnings
import numpy as np
from sklearn.kernel_ridge import KernelRidge
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
    kr = KernelRidge(kernel="rbf", gamma=20, alpha=0.01).fit(x[:, None], y)
    plt.scatter(x, y, s=8, alpha=0.5)
    plt.plot(xx, kr.predict(xx[:, None]), label="核岭回归 RBF")
    plt.legend()
    plt.title("正则化与再生核 (RKHS)")
    plt.savefig("07_Basis_Expansions_and_Regularization/7.7_Regularization_and_Reproducing_Kernel_Hilbert_Spaces/7.7_kridge.png", dpi=110)
    print("ok: 7.7")


if __name__ == "__main__":
    main()