# 4.8_Classes_of_Restricted_Estimators
"""受限估计器: 子集选择与岭收缩对比。"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.datasets import make_regression


def main() -> None:
    X, y = make_regression(n_samples=80, n_features=6, noise=0.4, random_state=0)
    full = LinearRegression().fit(X, y).coef_
    ridge = Ridge(alpha=10).fit(X, y).coef_
    print("满模型系数:", np.round(full, 2))
    print("岭回归系数:", np.round(ridge, 2))
    print("约束(收缩)使系数变小")


if __name__ == "__main__":
    main()