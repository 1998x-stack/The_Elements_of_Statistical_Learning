# 9.5_Estimates_of_In_Sample_Prediction_Error
"""样本内误差估计: Cp/AIC 与真实测试误差的对比。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 100
    X = rng.normal(size=(n, 5))
    beta = np.array([1.5, -1.0, 0.0, 0.0, 0.0])
    y = X @ beta + 0.6 * rng.standard_normal(n)
    lr = LinearRegression().fit(X, y)
    rss = np.sum((y - lr.predict(X)) ** 2)
    p = X.shape[1]
    err_tr = rss / n
    sigma2 = rss / (n - p - 1)
    cp = err_tr + 2 * p * sigma2 / n  # C_p 以 MSE 尺度
    Xte = rng.normal(size=(1000, 5))
    yte = Xte @ beta + 0.6 * rng.standard_normal(1000)
    test = np.mean((yte - lr.predict(Xte)) ** 2)
    print(f"C_p ≈ {cp:.3f}   真实测试误差 ≈ {test:.3f}")


if __name__ == "__main__":
    main()