# 11.1_Generalized_Additive_Models
"""广义可加模型: 用样条基对加性函数做可加拟合。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def basis1d(x, k=3, n_knots=6):
    x = np.asarray(x, dtype=float).ravel()
    knots = np.linspace(x.min(), x.max(), n_knots)
    cols = [np.ones_like(x), x, x ** 2, x ** 3]
    for kk in knots:
        cols.append(np.maximum(x - kk, 0) ** k)
    return np.column_stack(cols)


def main() -> None:
    n = 300
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(4 * x1) + np.cos(4 * x2) + 0.2 * rng.standard_normal(n)
    # 加性: 合并两维的样条基做线性回归 => 可加模型
    B1 = basis1d(x1)
    B2 = basis1d(x2)
    lr = LinearRegression().fit(np.hstack([B1, B2]), y)
    print("可加模型拟合, 特征数:", lr.n_features_in_)
    print("加性假设: y ≈ f1(x1)+f2(x2), 可解释。")


if __name__ == "__main__":
    main()