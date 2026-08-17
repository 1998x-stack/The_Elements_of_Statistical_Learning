# 5.11_Exercises
"""习题: 比较闭式岭回归与 sklearn Ridge, 验证一致性。"""
import warnings
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore", category=RuntimeWarning)


def ridge_closed(X, y, lam):
    Xb = np.c_[np.ones(len(X)), X]
    p = Xb.shape[1]
    A = Xb.T @ Xb + lam * np.eye(p)
    A[0, 0] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def main() -> None:
    X, y = load_diabetes(return_X_y=True)
    lam = 5.0
    beta_own = ridge_closed(X, y, lam)
    r = Ridge(alpha=lam, fit_intercept=True).fit(X, y)
    diff = np.abs(beta_own - np.r_[r.intercept_, r.coef_]).max()
    print("闭式解与 sklearn 的最大误差 =", diff)
    print("一致(接近0)?" , diff < 1e-6)


if __name__ == "__main__":
    main()