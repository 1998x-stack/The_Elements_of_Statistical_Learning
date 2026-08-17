# 10.9_Stochastic_Search:_Bumping
"""随机搜索 Bumping: 多次随机扰动选留出误差最小的模型。"""
import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 200
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(n)
    # 留一个子集选出 "幸运" 的模型
    Tr, Va = train_test_split(np.arange(n), test_size=0.3, random_state=0)
    best = (None, np.inf)
    for b in range(50):
        idx = rng.integers(Tr[0], Tr[-1] + 1, len(Tr))
        xc = x[idx]  # 在训练域内随机重抽
        m = DecisionTreeRegressor(max_depth=6, random_state=b).fit(xc[:, None], y[idx])
        err = mean_squared_error(y[Va], m.predict(x[Va, None]))
        if err < best[1]:
            best = (m, err)
    base = DecisionTreeRegressor(max_depth=6, random_state=42).fit(x[Tr, None], y[Tr])
    base_err = mean_squared_error(y[Va], base.predict(x[Va, None]))
    print(f"单次拟合验证MSE = {base_err:.4f}")
    print(f"Bumping(20次随机)验证MSE = {best[1]:.4f}")
    print('Bumping 通过“运气”挑选测试误差最优的扰动。')


if __name__ == "__main__":
    main()