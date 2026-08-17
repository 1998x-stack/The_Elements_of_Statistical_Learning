# 5.3_Subset_Selection
"""子集选择: 前向逐步加入特征并报告每步的拟合优度。"""
import warnings
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_diabetes(return_X_y=True)
    p = X.shape[1]
    selected = []
    for step in range(5):
        best = None
        for j in range(p):
            if j in selected:
                continue
            cols = selected + [j]
            r2 = LinearRegression().fit(X[:, cols], y).score(X[:, cols], y)
            if best is None or r2 > best[1]:
                best = (j, r2)
        selected.append(best[0])
        print(f"步 {step+1}: 加入特征 {selected[-1]:2d}  R^2={best[1]:.4f}")


if __name__ == "__main__":
    main()