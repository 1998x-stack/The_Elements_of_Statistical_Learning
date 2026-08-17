# 6.5_Separating_Hyperplanes
"""分隔超平面: 线性SVM(最大间隔)绘制边界。"""
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_blobs(n_samples=120, centers=2, cluster_std=1.2, random_state=0)
    clf = LinearSVC(C=1e6).fit(X, y)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
    z = clf.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, z, alpha=0.25, cmap="PRGn")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=12)
    plt.title("最大间隔分隔超平面")
    plt.savefig("06_Linear_Methods_for_Classification/6.5_Separating_Hyperplanes/6.5_boundary.png", dpi=110)
    print("分隔超平面系数:", np.round(clf.coef_[0], 3))


if __name__ == "__main__":
    main()