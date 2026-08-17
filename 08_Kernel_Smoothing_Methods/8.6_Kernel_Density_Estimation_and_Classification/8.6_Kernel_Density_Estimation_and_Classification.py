# 8.6_Kernel_Density_Estimation_and_Classification
"""核密度估计: 高斯核 KDE 与带宽影响。"""
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    x = np.concatenate([rng.normal(-1, 0.6, 400), rng.normal(1.2, 0.4, 400)])
    xx = np.linspace(-4, 4, 500)[:, None]
    plt.hist(x, bins=40, density=True, alpha=0.4, label="直方图")
    for bw in (0.05, 0.25):
        kd = KernelDensity(kernel="gaussian", bandwidth=bw).fit(x[:, None])
        plt.plot(xx, np.exp(kd.score_samples(xx)), label=f"KDE bw={bw}")
    plt.legend()
    plt.title("核密度估计 (带宽的影响)")
    plt.savefig("08_Kernel_Smoothing_Methods/8.6_Kernel_Density_Estimation_and_Classification/8.6_kde.png", dpi=110)
    print("ok: 8.6")


if __name__ == "__main__":
    main()