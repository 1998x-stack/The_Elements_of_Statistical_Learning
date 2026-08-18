# 16.5_Principal_Components,_Curves_and_Surfaces
"""主成分分析: 在鸢尾花上降维到 2D 并看解释方差。"""
import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    iris = load_iris()
    X = iris.data
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    plt.scatter(Z[:, 0], Z[:, 1], c=iris.target, cmap="Set1", s=20)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA 降维到 2D (鸢尾花)")
    plt.savefig("16_Unsupervised_Learning/16.5_Principal_Components,_Curves_and_Surfaces/16.5_pca.png", dpi=110)
    print("解释方差占比:", np.round(pca.explained_variance_ratio_, 3))


if __name__ == "__main__":
    main()