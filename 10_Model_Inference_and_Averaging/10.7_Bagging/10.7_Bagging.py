# 10.7_Bagging
"""Bagging: 对重采样后的决策树预测做平均, 降低方差。"""
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
    Xtr, Xte, ytr, yte = train_test_split(x[:, None], y, test_size=0.4, random_state=0)
    tree = DecisionTreeRegressor(max_depth=8, random_state=0).fit(Xtr, ytr)
    # bagging: B 棵自重采样训练树, 预测取平均
    B = 100
    preds = np.zeros((B, len(Xte)))
    for b in range(B):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        preds[b] = DecisionTreeRegressor(max_depth=8, random_state=b).fit(Xtr[idx], ytr[idx]).predict(Xte)
    bag = preds.mean(axis=0)
    print(f"单棵决策树测试MSE = {mean_squared_error(yte, tree.predict(Xte)):.4f}")
    print(f"Bagging(100棵树)测试MSE = {mean_squared_error(yte, bag):.4f}")
    print("Bagging 通常降低方差、改善测试误差。")


if __name__ == "__main__":
    main()