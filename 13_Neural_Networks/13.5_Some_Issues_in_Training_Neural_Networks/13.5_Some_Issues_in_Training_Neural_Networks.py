# 13.5_Some_Issues_in_Training_Neural_Networks
"""训练神经网络的注意点: 学习率/正则(alpha)与特征标准化。"""
import warnings
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
    Xstd = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xstd, y, test_size=0.3, random_state=0)
    for a in (1e-4, 1e-2, 1.0):
        m = MLPClassifier(hidden_layer_sizes=(10,), alpha=a, max_iter=500,
                          random_state=0).fit(Xtr, ytr)
        print(f"alpha={a:.4f}  测试准确率={accuracy_score(yte, m.predict(Xte)):.3f}")
    print("标准化输入 + 合适的 weight-decay(alpha) 影响收敛与过拟合。")


if __name__ == "__main__":
    main()