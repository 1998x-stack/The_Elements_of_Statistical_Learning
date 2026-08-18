# 13.3_Neural_Networks
"""神经网络: 单隐层 MLP 在二维合成数据上拟合并画决策面。"""
import warnings
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    X, y = make_moons(n_samples=300, noise=0.15, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation="tanh", max_iter=1000,
                        random_state=0).fit(Xtr, ytr)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - .2, X[:, 0].max() + .2, 250),
                         np.linspace(X[:, 1].min() - .2, X[:, 1].max() + .2, 250))
    z = mlp.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.contourf(gx, gy, z, alpha=0.3, cmap="bwr")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=12)
    plt.title("单隐层神经网络决策面")
    plt.savefig("13_Neural_Networks/13.3_Neural_Networks/13.3_nn.png", dpi=110)
    print("MLP 测试准确率 =", round(mlp.score(Xte, yte), 3))


if __name__ == "__main__":
    main()