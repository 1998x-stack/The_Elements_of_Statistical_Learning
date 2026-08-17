# 11.7_Computational_Considerations
"""计算考虑: 决策树 vs 核回归的拟合/预测耗时对比。"""
import time
import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 20000
    X = rng.normal(size=(n, 5))
    y = X @ np.array([1.0, -1.0, 0.5, 0.0, 0.0]) + rng.standard_normal(n)
    t0 = time.perf_counter()
    tree = DecisionTreeRegressor(max_depth=10, random_state=0).fit(X, y)
    t1 = time.perf_counter()
    lr = LinearRegression().fit(X, y)
    t2 = time.perf_counter()
    print(f"树拟合: {t1-t0:.4f}s   线性拟合: {t2-t1:.4f}s  (n={n}, p=5)")
    print("树在深数据上拟合较慢但预测快速; 线性用解析很快。")


if __name__ == "__main__":
    main()