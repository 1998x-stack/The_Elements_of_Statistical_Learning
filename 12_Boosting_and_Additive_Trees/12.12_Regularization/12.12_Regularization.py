# 12.12_Regularization
"""Boosting 正则化: 学习率 η 对误差的影响(收缩 = 早停等价)。"""
import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def run(eta, M):
    x = np.sort(rng.uniform(0, 1, 200))
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(200)
    Xtr, Xte, ytr, yte = train_test_split(x[:, None], y, test_size=0.4, random_state=0)
    Ftr = np.full_like(ytr, ytr.mean())
    Fte = np.full_like(yte, ytr.mean())
    for m in range(M):
        t = DecisionTreeRegressor(max_depth=2, random_state=m).fit(Xtr, ytr - Ftr)
        Ftr += eta * t.predict(Xtr)
        Fte += eta * t.predict(Xte)
    return mean_squared_error(yte, Fte)


def main() -> None:
    print("η      测试MSE(M=150)")
    for eta in (0.01, 0.1, 0.5):
        print(f"{eta:.2f}   {run(eta, 150):.4f}")
    print("小 η（收缩）+ 稍多树 通常更好, 等价于平滑化。")


if __name__ == "__main__":
    main()