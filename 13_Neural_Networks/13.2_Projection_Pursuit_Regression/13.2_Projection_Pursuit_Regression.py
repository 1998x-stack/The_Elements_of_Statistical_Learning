# 13.2_Projection_Pursuit_Regression
"""投影寻踪回归(PPR): 对投影后的指数平滑组合。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 300
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    # 真实依赖第一个投影(线性组合)的非线性函数
    y = np.sin(3 * (x1 - 0.5 * x2)) + 0.1 * rng.standard_normal(n)
    # 一投影方向的 PPR 思想: 投影 alpha = [1,-0.5] 后非线性(用平滑)
    # 这里用简单样条基拟合该一维投影 (近似 PPR 一轮)
    z = (x1 - 0.5 * x2)
    X = np.column_stack([z, z ** 2, z ** 3])
    lr = LinearRegression().fit(X, y)
    print("PPR 一轮: 平滑项拟合 R^2 =", round(lr.score(X, y), 3))
    print("说明: PPR/网络对投影变量做平滑后再组合。")


if __name__ == "__main__":
    main()