# 4.4_Statistical_Decision_Theory
"""监督学习的统计决策理论: 线性判别当作近贝叶斯分类器可视化。"""
import warnings
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_blobs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.2, random_state=0)
    lda = LinearDiscriminantAnalysis().fit(X, y)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))
    z = lda.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, z, alpha=0.3, cmap="bwr")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=12)
    plt.title("分类: LDA 作为贝叶斯分类器的近似")
    plt.savefig("04_Overview_of_Supervised_Learning/4.4_Statistical_Decision_Theory/4.4_贝叶斯分类器.png", dpi=110)
    print("ok: 4.4")


if __name__ == "__main__":
    main()