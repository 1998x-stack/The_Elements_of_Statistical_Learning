# 9.11_Bootstrap_Methods
"""自举方法: 用重采样估计线性回归系数或中位数的标准误。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def bootstrap_se(X, y, B=500):
    n = X.shape[0]
    betas = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        betas.append(LinearRegression().fit(X[idx], y[idx]).coef_)
    betas = np.array(betas)
    return betas.std(axis=0)


def main() -> None:
    n = 100
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.4 * rng.standard_normal(n)
    lr = LinearRegression().fit(X, y)
    se = bootstrap_se(X, y, 500)
    print("系数:", np.round(lr.coef_, 3))
    print("自举标准误(B=500):", np.round(se, 3))
    print("提示: 与解析标准误 σ̂√(X'X)^{-1} 接近。")


if __name__ == "__main__":
    main()