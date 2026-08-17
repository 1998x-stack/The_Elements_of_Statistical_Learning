# 5.8_More_on_the_Lasso_and_Related_Path_Algorithms
"""Lasso 路径: 随 alpha 变化系数逐步变为 0。"""
import warnings
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_regression(n_samples=60, n_features=8, n_informative=3, noise=0.4, random_state=0)
    for a in (1e-3, 0.05, 0.2, 0.6):
        l = Lasso(alpha=a, max_iter=10000).fit(X, y)
        print(f"alpha={a:.3f} 非零系数={int(np.sum(l.coef_ != 0))}")


if __name__ == "__main__":
    main()