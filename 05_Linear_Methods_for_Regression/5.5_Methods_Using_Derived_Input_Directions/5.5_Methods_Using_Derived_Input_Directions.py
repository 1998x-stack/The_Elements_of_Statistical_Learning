# 5.5_Methods_Using_Derived_Input_Directions
"""派生输入方向: PCR 与偏最小二乘(PLS)与最小二乘对比。"""
import warnings
import numpy as np
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_regression(n_samples=120, n_features=10, noise=0.6, random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ols = LinearRegression().fit(Xtr, ytr).predict(Xte)
    for n in (2, 5):
        pcr = PCA(n_components=n)
        Xp = pcr.fit_transform(Xtr)
        m = LinearRegression().fit(Xp, ytr)
        mse = mean_squared_error(yte, m.predict(pcr.transform(Xte)))
        print(f"PCR k={n} 测试MSE={mse:.2f}")
    pls = PLSRegression(n_components=5).fit(Xtr, ytr)
    print("PLS 测试MSE =", round(mean_squared_error(yte, pls.predict(Xte)), 2))
    print("OLS 测试MSE =", round(mean_squared_error(yte, ols), 2))


if __name__ == "__main__":
    main()