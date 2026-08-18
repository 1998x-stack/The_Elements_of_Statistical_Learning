# 12.14_Illustrations
"""boosting 可视化: 决策面随迭代数的改善。"""
import warnings
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_circles
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_boundary(model, X, y, title, fname):
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - .2, X[:, 0].max() + .2, 200),
                         np.linspace(X[:, 1].min() - .2, X[:, 1].max() + .2, 200))
    z = model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, z, alpha=0.3, cmap="bwr")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=10)
    plt.title(title)
    plt.savefig("12_Boosting_and_Additive_Trees/12.14_Illustrations/" + fname, dpi=110)
    plt.clf()


def main() -> None:
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=0)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=0).fit(X, y)
    plot_boundary(tree, X, y, "单棵决策树", "12.14_tree.png")
    plot_boundary(gbm, X, y, "GBM(50树)", "12.14_gbm.png")
    print("ok: 12.14 已生成两幅边界对比图")


if __name__ == "__main__":
    main()