# 3.4_The_Many_Faces_of_Regularization
"""正则化: 岭回归 vs 最小二乘在高阶多项式上的对比。"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(-3, 3, 80))
    y = np.sin(x) + 0.2 * rng.standard_normal(80)
    xx = np.linspace(-3, 3, 400)
    ls = make_pipeline(PolynomialFeatures(12), LinearRegression()).fit(x[:, None], y)
    ridge = make_pipeline(PolynomialFeatures(12), Ridge(alpha=5)).fit(x[:, None], y)
    plt.scatter(x, y, s=6, alpha=0.5)
    plt.plot(xx, np.sin(xx), "k--", label="真实")
    plt.plot(xx, ls.predict(xx[:, None]), label="最小二乘(过拟合)")
    plt.plot(xx, ridge.predict(xx[:, None]), label="岭回归")
    plt.legend()
    plt.title("正则化抑制过拟合")
    plt.savefig("03_Introduction/3.4_The_Many_Faces_of_Regularization/3.4_正则化.png", dpi=110)
    print("ok: 3.4")


if __name__ == "__main__":
    main()