# 7.10_Exercises
"""习题: 交叉验证比较样条与高次多项式在合成数据上的 R²。"""
import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def spline_basis(x, k=3, n_knots=5):
    x = np.asarray(x, dtype=float).ravel()
    knots = np.linspace(x.min(), x.max(), n_knots)
    cols = [np.ones_like(x)]
    for d in range(1, k + 1):
        cols.append(x ** d)
    for kk in knots:
        cols.append(np.maximum(x - kk, 0) ** k)
    return np.column_stack(cols)


class SplineRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.lr_ = LinearRegression().fit(spline_basis(X[:, 0]), y)
        return self

    def predict(self, X):
        return self.lr_.predict(spline_basis(X[:, 0]))


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 150))
    y = np.sin(5 * x) + 0.15 * rng.standard_normal(150)
    poly = make_pipeline(PolynomialFeatures(25), LinearRegression())
    for name, model in [("三次样条", SplineRegressor()), ("25次多项式", poly)]:
        r2 = cross_val_score(model, np.c_[x], y, cv=5).mean()
        print(f"{name}: 交叉验证 R^2 = {r2:.4f}")


if __name__ == "__main__":
    main()