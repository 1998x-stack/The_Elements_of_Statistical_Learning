# 11.2_Tree_Based_Methods
"""树方法: 分类回归树在二维合成数据上的边界与深度。"""
import warnings
import numpy as np
from sklearn.datasets import make_circles
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)


def main() -> None:
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=0)
    clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 300),
                         np.linspace(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, 300))
    z = clf.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, z, alpha=0.3, cmap="bwr")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=12)
    plt.title("决策树(深度4)的轴平行边界")
    plt.savefig("11_Additive_Models,_Trees,_and_Related_Methods/11.2_Tree_Based_Methods/11.2_tree.png", dpi=110)
    print("树深度:", clf.get_depth(), " 叶节点数:", clf.get_n_leaves())
    print(export_text(clf, feature_names=["x1", "x2"], max_depth=2))


if __name__ == "__main__":
    main()