# 4.3_Two_Simple_Approaches_to_Prediction:_Least_Squares_and_Nearest_Neighbors
"""两类简单预测方法: 最小二乘 vs k-近邻。"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(0, 4, 120))
    y = np.sin(1.5 * x) + 0.15 * rng.standard_normal(120)
    xx = np.linspace(0, 4, 400)
    lin = LinearRegression().fit(x[:, None], y)
    for k in (1, 15):
        knn = KNeighborsRegressor(n_neighbors=k).fit(x[:, None], y)
        plt.plot(xx, knn.predict(xx[:, None]), label=f"kNN k={k}")
    plt.plot(xx, lin.predict(xx[:, None]), "--", label="最小二乘")
    plt.plot(xx, np.sin(1.5 * xx), "k-", label="真实")
    plt.scatter(x, y, s=5, alpha=0.4)
    plt.legend()
    plt.title("最小二乘 vs K-近邻")
    plt.savefig("04_Overview_of_Supervised_Learning/4.3_Two_Simple_Approaches_to_Prediction:_Least_Squares_and_Nearest_Neighbors/4.3_LS_vs_kNN.png", dpi=110)
    print("ok: 4.3")


if __name__ == "__main__":
    main()