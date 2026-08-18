# 15.3_k_Nearest_Neighbor_Classifiers
"""k-最近邻分类器: 决策边界随 k 的变化。"""
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=0)
    for k in (1, 15, 60):
        knn = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 200),
                             np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 200))
        z = knn.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        plt.contourf(gx, gy, z, alpha=0.25, cmap="Pastel1")
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", s=12)
        plt.title(f"kNN k={k}")
        plt.savefig(f"15_Prototype_Methods_and_Nearest_Neighbors/15.3_k_Nearest_Neighbor_Classifiers/15.3_knn_{k}.png", dpi=110)
        plt.clf()
    knn = KNeighborsClassifier(n_neighbors=15).fit(X, y)
    print("k=15 训练准确率 =", round(knn.score(X, y), 3))


if __name__ == "__main__":
    main()