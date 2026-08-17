# 9.10_Cross_Validation
"""交叉验证: K折 CV 在模型复杂度(GAM-样条/多项式)上的误差曲线。"""
import warnings
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(-3, 3, 200))
    y = np.sin(x) + 0.2 * rng.standard_normal(200)
    print("K    5折CV误差")
    for K in (2, 5, 10, 20, 40, 200):
        scores = cross_val_score(make_pipeline(PolynomialFeatures(4), LinearRegression()), x[:, None], y,
                                 cv=K, scoring="neg_mean_squared_error")
        mse = -scores.mean()
        if K % 5 == 0 or K == 200:
            print(f"{K:4d}   {mse:.4f}")
    # 模型复杂度上的 CV(9.2 的 U 型)
    print("度数 5折CV误差")
    for d in (1, 3, 5, 9):
        m = make_pipeline(PolynomialFeatures(d), LinearRegression())
        mse = -cross_val_score(m, x[:, None], y, cv=5, scoring="neg_mean_squared_error").mean()
        print(f"{d:3d}   {mse:.4f}")


if __name__ == "__main__":
    main()