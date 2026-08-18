# 12.10_Numerical_Optimization_via_Gradient_Boosting
"""梯度提升: 每轮对负梯度拟合小树并加学习率缩放。"""
import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 200
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(n)
    Xtr, Xte, ytr, yte = train_test_split(x[:, None], y, test_size=0.4, random_state=0)
    eta = 0.1
    Ftr = np.full_like(ytr, ytr.mean())
    Fte = np.full_like(yte, ytr.mean())
    for m in range(200):
        neg_grad = ytr - Ftr            # 平方损失的负梯度=残差
        t = DecisionTreeRegressor(max_depth=3, random_state=m).fit(Xtr, neg_grad)
        Ftr += eta * t.predict(Xtr)
        Fte += eta * t.predict(Xte)
        if m + 1 in (1, 20, 100, 200):
            print(f"M={m+1:4d} 测试MSE={mean_squared_error(yte, Fte):.4f}")
    print("梯度提升: 每轮拟合负梯度(残差), 用学习率 η=%.2f 收缩。" % eta)


if __name__ == "__main__":
    main()