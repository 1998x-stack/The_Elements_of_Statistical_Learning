# 6.3_Linear_Discriminant_Analysis
"""线性判别分析 LDA: 决策边界与判别函数。"""
import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    iris = load_iris()
    X = iris.data[:, :2]  # 用两个特征便于可视化
    y = iris.target
    lda = LinearDiscriminantAnalysis().fit(X, y)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
    Z = lda.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, Z, alpha=0.3, cmap="Pastel1")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", s=15)
    plt.title("LDA 决策区域 (鸢尾花前两特征)")
    plt.savefig("06_Linear_Methods_for_Classification/6.3_Linear_Discriminant_Analysis/6.3_LDA边界.png", dpi=110)
    print("训练准确率:", round(lda.score(X, y), 3))


if __name__ == "__main__":
    main()