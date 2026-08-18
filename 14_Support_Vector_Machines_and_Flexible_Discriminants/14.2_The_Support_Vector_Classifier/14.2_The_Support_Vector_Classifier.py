# 14.2_The_Support_Vector_Classifier
"""支持向量分类器: 最大间隔 + 软间隔(C 控制)。"""
import warnings
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    X, y = make_blobs(n_samples=120, centers=2, cluster_std=1.5, random_state=0)
    clf = LinearSVC(C=1.0).fit(X, y)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 250),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 250))
    z = clf.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, z, alpha=0.3, cmap="bwr")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=12)
    plt.title("支持向量分类器 (软间隔, C=1)")
    plt.savefig("14_Support_Vector_Machines_and_Flexible_Discriminants/14.2_The_Support_Vector_Classifier/14.2_boundary.png", dpi=110)
    print("ok: 14.2 线性软间隔超平面")


if __name__ == "__main__":
    main()