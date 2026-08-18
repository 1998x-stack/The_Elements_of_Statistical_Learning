# 14.3_Support_Vector_Machines_and_Kernels
"""核支持向量机: 在非线性(双月)数据上用 RBF/多项式核。"""
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    X, y = make_moons(n_samples=250, noise=0.12, random_state=0)
    svc = SVC(kernel="rbf", gamma=2).fit(X, y)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - .2, X[:, 0].max() + .2, 250),
                         np.linspace(X[:, 1].min() - .2, X[:, 1].max() + .2, 250))
    z = svc.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, z, alpha=0.3, cmap="bwr")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=12)
    plt.title("RBF 核 SVM (双月)")
    plt.savefig("14_Support_Vector_Machines_and_Flexible_Discriminants/14.3_Support_Vector_Machines_and_Kernels/14.3_kernel.png", dpi=110)
    print("RBF-SVM 训练准确率 =", round(svc.score(X, y), 3))


if __name__ == "__main__":
    main()