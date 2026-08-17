# 4.7_Structured_Regression_Models
"""结构化回归: 线性/加性假设 vs 一般局部方法, 用双输入演示。"""
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)


def main() -> None:
    n = 200
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = 3 * x1 - 2 * x2 + 0.1 * rng.standard_normal(n)
    lr = LinearRegression().fit(np.c_[x1, x2], y)
    print("拟合参数(真实为 [3, -2]):", np.round(lr.coef_, 3))


if __name__ == "__main__":
    main()