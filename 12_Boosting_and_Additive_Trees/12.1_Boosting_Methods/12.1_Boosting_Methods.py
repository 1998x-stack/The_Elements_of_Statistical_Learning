# 12.1_Boosting_Methods
"""Boosting 概述: 手写 AdaBoost(指数损失+样本权重)在月亮数据集上。"""
import warnings
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
    y = np.where(y == 1, 1, -1)
    M = 60
    w = np.ones(len(X)) / len(X)
    alphas = []
    trees = []
    for m in range(M):
        t = DecisionTreeClassifier(max_depth=1, random_state=m).fit(X, y, sample_weight=w)
        pred = t.predict(X)
        err = np.sum(w * (pred != y)) / np.sum(w)
        alpha = 0.5 * np.log((1 - err) / (err + 1e-12))
        w = w * np.exp(-alpha * y * pred)
        w /= w.sum()
        alphas.append(alpha)
        trees.append(t)
    # 集成预测
    F = sum(a * t.predict(X) for a, t in zip(alphas, trees))
    train_err = np.mean(np.sign(F) != y)
    print(f"AdaBoost M={M}: 训练误差 = {train_err:.4f}")
    print("每轮对弱学习器(桩)重新加权, 集成后预测更强。")


if __name__ == "__main__":
    main()