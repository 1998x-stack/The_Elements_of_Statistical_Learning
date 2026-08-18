# 13.4_Fitting_Neural_Networks
"""拟合神经网络: 用训练误差曲线观察 MLP 随迭代下降。"""
import warnings
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    X, y = make_moons(n_samples=300, noise=0.15, random_state=0)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), activation="tanh", max_iter=300,
                        warm_start=False, random_state=0)
    mlp.fit(X, y)
    print("最终 loss:", round(float(mlp.loss_), 4))
    print("迭代次数:", mlp.n_iter_, "  达到 tol 或 max_iter")
    print("拟合通过反向传播+梯度下降最小化平方/交叉熵误差。")


if __name__ == "__main__":
    main()