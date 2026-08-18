# 13.9_Bayesian_Neural_Nets_and_the_NIPS_2003_Challenge
"""贝叶斯神经网络: 用"多个随机初始化的平均"近似预测不确定性/提升稳健。"""
import warnings
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
    preds = []
    for seed in range(20):
        m = MLPClassifier(hidden_layer_sizes=(10,), max_iter=400, random_state=seed).fit(X, y)
        preds.append(m.predict_proba(X)[:, 1])
    P = np.mean(preds, axis=0)          # 平均化(近似积分后验)
    acc = np.mean((P > 0.5) == y)
    print("20 个随机初始化平均的准确率 =", round(acc, 3))
    print("提示: 平均/集成可看作贝叶斯后验的近似(见 NIPS 2003 挑战)。")


if __name__ == "__main__":
    main()