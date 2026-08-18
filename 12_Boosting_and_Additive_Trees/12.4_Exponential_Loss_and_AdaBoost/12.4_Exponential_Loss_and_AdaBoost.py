# 12.4_Exponential_Loss_and_AdaBoost
"""指数损失与 AdaBoost: 权重更新与每轮加权误差。"""
import warnings
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
    y = np.where(y == 1, 1, -1)
    M = 40
    w = np.ones(len(X)) / len(X)
    preds = []
    alphas = []
    for m in range(M):
        t = DecisionTreeClassifier(max_depth=1, random_state=m).fit(X, y, sample_weight=w)
        p = t.predict(X)
        err = np.sum(w * (p != y)) / np.sum(w)
        a = 0.5 * np.log((1 - err) / (err + 1e-12))
        w = w * np.exp(-a * y * p)
        w /= w.sum()
        preds.append(p)
        alphas.append(a)
    F = sum(a * p for a, p in zip(alphas, preds))
    print("指数损失样本权重 = w·exp(-y f(x)); 集成 0-1 误差 =",
          round(np.mean(np.sign(F) != y), 4))
    print("AdaBoost = 指数损失下的 FSAM 解: 每轮重新加权线性组合。")


if __name__ == "__main__":
    main()