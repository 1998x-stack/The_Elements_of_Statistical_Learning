# 4.6_Statistical_Models,_Supervised_Learning_and_Function_Approximation
"""函数逼近视角: 多项式拟合的均方误差随阶数的变化。"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(-3, 3, 150))
    y = np.sin(x) + 0.2 * rng.standard_normal(150)
    for deg in (1, 3, 7):
        model = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(x[:, None], y)
        mse = mean_squared_error(y, model.predict(x[:, None]))
        print(f"阶数 p={deg}: 训练MSE={mse:.3f}")


if __name__ == "__main__":
    main()