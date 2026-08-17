# 5.9_Computational_Considerations
"""计算注意: 比较正规方程与 QR/最小二乘求解耗时。"""
import time
import numpy as np
from sklearn.datasets import make_regression

rng = np.random.default_rng(0)


def main() -> None:
    X, y = make_regression(n_samples=1500, n_features=400, noise=0.1, random_state=0)
    t0 = time.perf_counter()
    np.linalg.solve(X.T @ X, X.T @ y)
    t1 = time.perf_counter()
    np.linalg.lstsq(X, y, rcond=None)
    t2 = time.perf_counter()
    print(f"正规方程(X'X): {t1-t0:.4f}s    lstsq(QR): {t2-t1:.4f}s")


if __name__ == "__main__":
    main()