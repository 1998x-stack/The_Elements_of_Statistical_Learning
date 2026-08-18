# 12.9_Boosting_Trees
"""Boosting 树: 依次对残差拟合小树并累加, 报告不同 M 的测试MSE。"""
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
    Ftr = np.zeros_like(ytr)
    Fte = np.zeros_like(yte)
    trees = []
    for m in range(200):
        t = DecisionTreeRegressor(max_depth=2, random_state=m).fit(Xtr, ytr - Ftr)
        Ftr += t.predict(Xtr)
        Fte += t.predict(Xte)
        trees.append(t)
        if m + 1 in (1, 10, 50, 100, 200):
            print(f"M={m+1:4d}  测试MSE={mean_squared_error(yte, Fte):.4f}")
    print("Boosting 树: 每轮拟合残差(小树), 累加即集成。")


if __name__ == "__main__":
    main()