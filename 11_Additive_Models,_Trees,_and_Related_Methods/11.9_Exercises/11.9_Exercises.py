# 11.9_Exercises
"""习题: 比较线性回归、决策树与 GAM 组合(加性)在合成数据的测试误差。"""
import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 300
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(4 * x1) + np.cos(4 * x2) + 0.2 * rng.standard_normal(n)
    X = np.c_[x1, x2]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
    models = {
        "线性": LinearRegression().fit(Xtr, ytr),
        "决策树": DecisionTreeRegressor(max_depth=5, random_state=0).fit(Xtr, ytr),
        "深决策树": DecisionTreeRegressor(max_depth=None, random_state=0).fit(Xtr, ytr),
    }
    for name, m in models.items():
        print(f"{name}: 测试MSE={mean_squared_error(yte, m.predict(Xte)):.4f}")


if __name__ == "__main__":
    main()