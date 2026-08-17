# 4.9_Model_Selection_and_the_Bias_Variance_Tradeoff
"""偏差-方差分解模拟: 最近邻, 不同 k。"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

rng = np.random.default_rng(0)


def f(x):
    return np.sin(1.5 * x)


def main() -> None:
    n, rep = 120, 200
    ks = [1, 3, 9, 30, 80]
    xs = np.linspace(0, 4, 150)
    true = f(xs)
    print("k      bias^2   var")
    for k in ks:
        preds = np.zeros((rep, len(xs)))
        for r in range(rep):
            x = rng.uniform(0, 4, n)
            y = f(x) + 0.2 * rng.standard_normal(n)
            preds[r] = KNeighborsRegressor(n_neighbors=k).fit(x[:, None], y).predict(xs[:, None])
        b = np.mean((preds.mean(0) - true) ** 2)
        v = np.mean(np.var(preds, axis=0))
        print(f"{k:3d}    {b:.3f}  {v:.3f}")


if __name__ == "__main__":
    main()