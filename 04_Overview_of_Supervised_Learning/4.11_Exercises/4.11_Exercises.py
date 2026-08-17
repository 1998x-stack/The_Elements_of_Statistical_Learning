# 4.11_Exercises
"""习题: 用留出集选择 kNN 的 k, 比较不同 k 的测试误差。"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(0, 4, 300))
    y = np.sin(1.5 * x) + 0.2 * rng.standard_normal(300)
    tr, te = x[:200], x[200:]
    ytr, yte = y[:200], y[200:]
    best = None
    for k in (1, 3, 5, 10, 20, 40):
        m = KNeighborsRegressor(n_neighbors=k).fit(tr[:, None], ytr)
        err = mean_squared_error(yte, m.predict(te[:, None]))
        print(f"k={k:3d} 测试MSE={err:.4f}")
        if best is None or err < best[1]:
            best = (k, err)
    print("最优 k =", best[0], "MSE=", round(best[1], 4))


if __name__ == "__main__":
    main()