# 9.4_Optimism_of_the_Training_Error_Rate
"""训练误差的乐观性: 训练误差 vs 测试误差的差距(乐观度)。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 100
    X = rng.normal(size=(n, 3))
    beta = np.array([1.0, -2.0, 0.5])
    y = X @ beta + 0.5 * rng.standard_normal(n)
    lr = LinearRegression().fit(X, y)
    pred_tr = lr.predict(X)
    err_train = mean_squared_error(y, pred_tr)
    # 生成独立新样本估计通用误差
    Xte = rng.normal(size=(500, 3))
    yte = Xte @ beta + 0.5 * rng.standard_normal(500)
    err_test = mean_squared_error(yte, lr.predict(Xte))
    optimism = err_test - err_train
    print(f"训练误差={err_train:.3f}   泛化误差={err_test:.3f}   乐观度={optimism:.3f}")


if __name__ == "__main__":
    main()