# 7.10_Exercises
"""习题: 交叉验证比较三次样条与中高阶多项式在合成数据上的 R²。"""
import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


class SplineRegressor(BaseEstimator, RegressorMixin):
    """三次样条回归(截断幂基). 训练时记录节点, 预测时复用同一节点."""

    def __init__(self, k=3, n_knots=5):
        self.k, self.n_knots = k, n_knots

    def _basis(self, x):
        x = np.asarray(x, dtype=float).ravel()
        cols = [np.ones_like(x), x, x ** 2, x ** 3]
        for kk in self.knots_:
            cols.append(np.maximum(x - kk, 0) ** self.k)
        return np.column_stack(cols)

    def fit(self, X, y):
        x = np.asarray(X[:, 0], dtype=float)
        self.knots_ = np.linspace(x.min(), x.max(), self.n_knots)
        self.lr_ = LinearRegression().fit(self._basis(x), y)
        return self

    def predict(self, X):
        return self.lr_.predict(self._basis(X[:, 0]))


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 150))
    y = np.sin(5 * x) + 0.15 * rng.standard_normal(150)
    models = [
        ("三次样条", SplineRegressor(k=3, n_knots=5)),
        ("9次多项式", make_pipeline(PolynomialFeatures(9), LinearRegression())),
        ("线性", make_pipeline(PolynomialFeatures(1), LinearRegression(include_bias=False))),
    ]
    for name, model in models:
        r2 = cross_val_score(model, np.c_[x], y, cv=5).mean()
        print(f"{name}: 交叉验证 R^2 = {r2:.4f}")


if __name__ == "__main__":
    main()