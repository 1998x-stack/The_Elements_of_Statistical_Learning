# 12.16_Exercises
"""习题: 对比不同 α/GBN 的超参数对测试误差影响。"""
import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def gbm(depth, eta, M):
    x = np.sort(rng.uniform(0, 1, 200))
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(200)
    Xtr, Xte, ytr, yte = train_test_split(x[:, None], y, test_size=0.4, random_state=0)
    Ftr = np.full_like(ytr, ytr.mean())
    Fte = np.full_like(yte, ytr.mean())
    for m in range(M):
        t = DecisionTreeRegressor(max_depth=depth, random_state=m).fit(Xtr, ytr - Ftr)
        Ftr += eta * t.predict(Xtr)
        Fte += eta * t.predict(Xte)
    return mean_squared_error(yte, Fte)


def main() -> None:
    print("超参数(树深,η,M)  测试MSE")
    print(f"  deep=2 η=0.1 M=100 → {gbm(2, 0.1, 100):.4f}")
    print(f"  deep=2 η=0.2 M=100 → {gbm(2, 0.2, 100):.4f}")
    print(f"  deep=3 η=0.1 M=200 → {gbm(3, 0.1, 200):.4f}")
    print("小深、小学习率、适当树数往往最优。")


if __name__ == "__main__":
    main()