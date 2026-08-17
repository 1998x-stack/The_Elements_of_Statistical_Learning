# 10.2_The_Bootstrap_and_Maximum_Likelihood_Methods
"""自举与极大似然: 比较系数标准误的 bootstrap 与解析(MLE)估计。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 100
    X = rng.normal(size=(n, 2))
    y = X @ np.array([1.0, -2.0]) + 0.5 * rng.standard_normal(n)
    lr = LinearRegression().fit(X, y)
    rss = np.sum((y - lr.predict(X)) ** 2)
    sigma2 = rss / (n - 3)
    Xb = np.c_[np.ones(n), X]
    cov = sigma2 * np.linalg.inv(Xb.T @ Xb)
    mle_se = np.sqrt(np.diag(cov)[1:])
    # bootstrap
    B = 500
    coefs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        coefs.append(LinearRegression().fit(X[idx], y[idx]).coef_)
    boot_se = np.std(coefs, axis=0)
    print("MLE 标准误:", np.round(mle_se, 3))
    print("Bootstrap 标准误(B=500):", np.round(boot_se, 3))
    print("两者接近, 体现自举与 MLE 推断的一致性。")


if __name__ == "__main__":
    main()