# 6.4_Logistic_Regression
"""逻辑回归: 拟合二分类并绘制决策边界。"""
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_blobs(n_samples=300, centers=2, cluster_std=2.0, random_state=0)
    clf = LogisticRegression().fit(X, y)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 300),
                         np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 300))
    z = clf.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, z, alpha=0.3, cmap="bwr")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=12)
    plt.title("逻辑回归决策边界")
    plt.savefig("06_Linear_Methods_for_Classification/6.4_Logistic_Regression/6.4_决策边界.png", dpi=110)
    print("系数:", np.round(clf.intercept_[0], 2), np.round(clf.coef_[0], 2))
    print("训练准确率:", round(clf.score(X, y), 3))


if __name__ == "__main__":
    main()