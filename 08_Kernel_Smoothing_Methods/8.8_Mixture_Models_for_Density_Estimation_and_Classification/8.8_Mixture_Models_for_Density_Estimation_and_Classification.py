# 8.8_Mixture_Models_for_Density_Estimation_and_Classification
"""混合模型: 高斯混合(GMM)估计密度并用于分类。"""
import warnings
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    X = np.vstack([rng.multivariate_normal([-2, 0], [[0.5, 0], [0, 0.5]], 300),
                   rng.multivariate_normal([2, 0], [[0.5, 0], [0, 0.5]], 300)])
    gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
    labels = gmm.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="bwr", s=10)
    for mu in gmm.means_:
        plt.scatter(*mu, marker="x", s=200, c="k")
    plt.title("高斯混合模型 (EM)")
    plt.savefig("08_Kernel_Smoothing_Methods/8.8_Mixture_Models_for_Density_Estimation_and_Classification/8.8_gmm.png", dpi=110)
    print("ok: 8.8 估计的两类中心:", np.round(gmm.means_, 2))


if __name__ == "__main__":
    main()