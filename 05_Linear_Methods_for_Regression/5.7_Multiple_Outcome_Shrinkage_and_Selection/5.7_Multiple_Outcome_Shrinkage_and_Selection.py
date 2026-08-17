# 5.7_Multiple_Outcome_Shrinkage_and_Selection
"""多元输出的收缩: 岭回归处理相关响应。"""
import warnings
import numpy as np
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    X = rng.normal(size=(100, 3))
    Y = np.c_[X[:, 0] + X[:, 1], X[:, 1] - X[:, 0]] + rng.normal(scale=0.2, size=(100, 2))
    r = Ridge(alpha=5).fit(X, Y)
    print("多输出岭回归系数形状:", r.coef_.shape)
    print("第一输出系数:", np.round(r.coef_[0], 3))


if __name__ == "__main__":
    main()