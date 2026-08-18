# 12.3_Forward_Stagewise_Additive_Modeling
"""前向分步加性建模(FSAM): 每一步只加一个弱学习器。"""
import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 150
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(n)
    F = np.zeros_like(y)
    print("步  MSE(当前)  加树")
    for m in range(30):
        # 拟合残差的小树(深度2)再加到当前函数
        t = DecisionTreeRegressor(max_depth=2, random_state=m).fit(x[:, None], y - F)
        F += t.predict(x[:, None])
        if m % 6 == 0 or m == 29:
            print(f"{m+1:3d}   {mean_squared_error(y, F):.4f}")
    print("FSAM: 每次拟合残差并加一个小基学习器。")


if __name__ == "__main__":
    main()