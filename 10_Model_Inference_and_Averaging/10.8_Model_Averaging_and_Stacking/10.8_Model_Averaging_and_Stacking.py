# 10.8_Model_Averaging_and_Stacking
"""模型平均与 Stacking: 加权/元层组合多个基学习器。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 300))[:, None]
    y = np.sin(6 * x.ravel()) + 0.2 * rng.standard_normal(300)
    Xtr, Xte, ytr, yte = train_test_split(x, y, test_size=0.4, random_state=0)
    base = {
        "树": DecisionTreeRegressor(max_depth=5, random_state=0).fit(Xtr, ytr),
        "kNN": KNeighborsRegressor(n_neighbors=15).fit(Xtr, ytr),
        "线性": LinearRegression().fit(Xtr, ytr),
    }
    # 简单平均
    avg = np.mean([m.predict(Xte) for m in base.values()], axis=0)
    # stacking: 用测试几基预测作为元特征拟元线性层
    meta_tr = np.column_stack([m.predict(Xtr) for m in base.values()])
    meta_te = np.column_stack([m.predict(Xte) for m in base.values()])
    stack = LinearRegression().fit(meta_tr, ytr).predict(meta_te)
    for name, m in base.items():
        print(f"{name}: {mean_squared_error(yte, m.predict(Xte)):.4f}")
    print(f"平均组合 : {mean_squared_error(yte, avg):.4f}")
    print(f"Stacking: {mean_squared_error(yte, stack):.4f}")


if __name__ == "__main__":
    main()