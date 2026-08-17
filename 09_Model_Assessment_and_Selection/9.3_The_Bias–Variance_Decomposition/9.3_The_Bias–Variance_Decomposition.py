# 9.3_The_Bias_Variance_Decomposition
"""偏差-方差分解: 最近邻回归的分解随 k 变化。"""
import warnings
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def f(x):
    return np.sin(2 * np.pi * x)


def main() -> None:
    n, rep = 150, 200
    xs = np.linspace(0, 1, 120)
    true = f(xs)
    print("k   bias^2   var     MSE(sim)")
    for k in (1, 5, 20, 80):
        P = np.zeros((rep, len(xs)))
        for r in range(rep):
            x = rng.uniform(0, 1, n)
            y = f(x) + 0.15 * rng.standard_normal(n)
            P[r] = KNeighborsRegressor(n_neighbors=k).fit(x[:, None], y).predict(xs[:, None])
        bias2 = np.mean((P.mean(0) - true) ** 2)
        var = np.mean(np.var(P, axis=0))
        mse = np.mean((P - true) ** 2)
        print(f"{k:3d}   {bias2:.3f}  {var:.3f}  {mse:.3f}")
    print("提示: MSE≈bias^2+var+σ²")


if __name__ == "__main__":
    main()