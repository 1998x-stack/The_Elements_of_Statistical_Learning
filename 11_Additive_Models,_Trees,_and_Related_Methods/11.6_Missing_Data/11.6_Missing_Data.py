# 11.6_Missing_Data
"""缺失数据: 简单均值插补 vs 完整数据的回归性能。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 200
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.4 * rng.standard_normal(n)
    # 制造缺失(MAR 简化模拟)
    Xm = X.copy()
    miss = rng.random(n) < 0.3
    Xm[miss, 1] = np.nan
    # 均值插补
    imp = SimpleImputer(strategy="mean")
    Ximp = imp.fit_transform(Xm)
    lr_full = LinearRegression().fit(X, y)
    lr_imp = LinearRegression().fit(Ximp, y)
    print("无缺失数据系数:", np.round(lr_full.coef_, 2))
    print("均值插补后系数:", np.round(lr_imp.coef_, 2))
    print("提示: 均值插补会缩小方差、可能带来偏差；/完整数据最优。")


if __name__ == "__main__":
    main()