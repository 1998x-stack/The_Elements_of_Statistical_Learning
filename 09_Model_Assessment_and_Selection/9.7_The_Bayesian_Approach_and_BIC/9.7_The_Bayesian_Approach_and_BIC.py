# 9.7_The_Bayesian_Approach_and_BIC
"""贝叶斯方法与 BIC: 对不同规模线性模型计算 BIC。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def bic(y, yhat, d, n):
    rss = np.sum((y - yhat) ** 2)
    return n * np.log(rss / n) + d * np.log(n)


def main() -> None:
    n = 200
    X = rng.normal(size=(n, 5))
    y = 1.5 * X[:, 0] - X[:, 1] + 0.5 * rng.standard_normal(n)
    print("模型规模  BIC")
    for d in (1, 2, 3, 5):
        lr = LinearRegression().fit(X[:, :d], y)
        b = bic(y, lr.predict(X[:, :d]), d + 1, n)
        print(f"   {d}     {b:.2f}")


if __name__ == "__main__":
    main()