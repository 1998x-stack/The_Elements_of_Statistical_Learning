# 3.2_Supervised_and_Unsupervised_Learning
"""无监督学习示例: K-均值聚类。"""
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)


def main() -> None:
    centers = [(0, 0), (4, 0), (2, 3)]
    X = np.vstack([rng.multivariate_normal(c, 0.4 * np.eye(2), 100) for c in centers])
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="viridis", s=12)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color="red", marker="x", s=120)
    plt.title("无监督学习: K-means 聚类 (K=3)")
    plt.savefig("03_Introduction/3.2_Supervised_and_Unsupervised_Learning/3.2_KMeans.png", dpi=110)
    print("ok: inertia =", round(km.inertia_, 1))


if __name__ == "__main__":
    main()