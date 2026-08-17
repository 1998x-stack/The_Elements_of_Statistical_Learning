# 9.6_The_Effective_Number_of_Parameters
"""有效参数数: 岭回归 df=tr(S_λ) 随 λ 变化。"""
import warnings
import numpy as np
from sklearn.datasets import make_regression
warnings.filterwarnings("ignore", category=RuntimeWarning)


def ridge_df(X, lam):
    p = X.shape[1]
    return np.trace(X @ np.linalg.inv(X.T @ X + lam * np.eye(p)) @ X.T)


def main() -> None:
    X, _ = make_regression(n_samples=80, n_features=6, random_state=0)
    print("lambda   有效自由度 df")
    for lam in (0.0, 0.5, 5.0, 50.0, 500.0):
        print(f"{lam:7g}   {ridge_df(X, lam):.3f}")


if __name__ == "__main__":
    main()