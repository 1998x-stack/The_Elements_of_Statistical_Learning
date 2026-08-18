# 12.6_Loss_Functions_and_Robustness
"""损失函数与鲁棒性: 平方/绝对/Huber 在含离群点回归上。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def mae(a, b):
    return np.mean(np.abs(a - b))


def main() -> None:
    n = 200
    x = np.linspace(0, 1, n)
    y = 3 * x + rng.normal(0, 0.2, n)
    y[::20] += 5  # 加离群点
    lr = LinearRegression().fit(x[:, None], y)
    hub = HuberRegressor(epsilon=1.35).fit(x[:, None], y)
    print(f"最小二乘 MAE={mae(y, lr.predict(x[:,None])):.3f}  系数={lr.coef_[0]:.2f}")
    print(f"Huber      MAE={mae(y, hub.predict(x[:,None])):.3f}  系数={hub.coef_[0]:.2f}")
    print("Huber (鲁棒) 比平方损失对离群点更稳, 更接近真实斜率 3。")


if __name__ == "__main__":
    main()