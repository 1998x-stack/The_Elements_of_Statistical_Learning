# 10.11_Exercises
"""习题: 比较 bagging 与单棵决策树随方差的变化。"""
import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def bagged_err(B, Xtr, ytr, Xte, yte):
    preds = np.zeros((B, len(Xte)))
    for b in range(B):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        preds[b] = DecisionTreeRegressor(max_depth=8, random_state=b).fit(Xtr[idx], ytr[idx]).predict(Xte)
    return mean_squared_error(yte, preds.mean(0))


def main() -> None:
    n = 200
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(n)
    Xtr, Xte, ytr, yte = train_test_split(x[:, None], y, test_size=0.4, random_state=0)
    print("B   bagging测试MSE")
    for B in (1, 5, 20, 100):
        print(f"{B:4d}   {bagged_err(B, Xtr, ytr, Xte, yte):.4f}")


if __name__ == "__main__":
    main()