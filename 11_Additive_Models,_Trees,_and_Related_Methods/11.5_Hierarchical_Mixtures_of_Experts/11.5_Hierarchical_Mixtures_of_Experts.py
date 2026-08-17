# 11.5_Hierarchical_Mixtures_of_Experts
"""分层专家混合: 门控软划分两个专家线性模型。"""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 300
    x = np.sort(rng.uniform(0, 1, n))
    # 两个专家: 低区线性 y≈1, 高区 y≈3
    y_true = np.where(x < 0.5, 1 + 0.5 * x, 3 - 0.5 * x)
    y = y_true + 0.05 * rng.standard_normal(n)
    # 用 (softmax) 门控近似——这里用加权分段拟合示意
    e1 = LinearRegression().fit(x[x < 0.5, None], y[x < 0.5])
    e2 = LinearRegression().fit(x[x >= 0.5, None], y[x >= 0.5])
    print("专家1(低区)x系数:", np.round(e1.coef_[0], 3), " 专家2(高区):", np.round(e2.coef_[0], 3))


if __name__ == "__main__":
    main()