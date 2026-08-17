# 5.2_Linear_Regression_Models_and_Least_Squares
"""最小二乘线性回归: 正规方程与矩阵求解, 附残差图。"""
import warnings
import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_diabetes(return_X_y=True)
    Xb = np.c_[np.ones(len(X)), X]
    # 正规方程 beta = (X'X)^-1 X'y, 用 solve 避免显式求逆
    beta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
    yhat = Xb @ beta
    rss = np.sum((y - yhat) ** 2)
    print("闭式解 beta0 = {:.3f}".format(beta[0]))
    print("RSS =", round(rss, 2))
    plt.scatter(yhat, y - yhat, s=10)
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("拟合值")
    plt.ylabel("残差")
    plt.title("残差 vs 拟合值 (糖尿病数据)")
    plt.savefig("05_Linear_Methods_for_Regression/5.2_Linear_Regression_Models_and_Least_Squares/5.2_残差图.png", dpi=110)
    print("ok: 5.2")


if __name__ == "__main__":
    main()