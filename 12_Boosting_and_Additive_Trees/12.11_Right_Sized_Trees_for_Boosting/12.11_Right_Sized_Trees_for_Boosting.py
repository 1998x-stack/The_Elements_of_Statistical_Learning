# 12.11_Right_Sized_Trees_for_Boosting
"""Boosting 树的合适大小: 不同树深度对测试误差的影响。"""
import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def gbm(depth, M=100):
    x = np.sort(rng.uniform(0, 1, 200))
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(200)
    Xtr, Xte, ytr, yte = train_test_split(x[:, None], y, test_size=0.4, random_state=0)
    Ftr = np.full_like(ytr, ytr.mean())
    Fte = np.full_like(yte, ytr.mean())
    eta = 0.1
    for m in range(M):
        t = DecisionTreeRegressor(max_depth=depth, random_state=m).fit(Xtr, ytr - Ftr)
        Ftr += eta * t.predict(Xtr)
        Fte += eta * t.predict(Xte)
    return mean_squared_error(yte, Fte)


def main() -> None:
    print("树深度  测试MSE(M=100)")
    for d in (1, 2, 4, 8):
        print(f"  {d:3d}   {gbm(d):.4f}")
    print("通常小树(深度2-4)作基学习器的 boosting 效果最好。")


if __name__ == "__main__":
    main()