# 7.2_Piecewise_Polynomials_and_Splines
"""分段多项式与样条: 三次样条 vs 全局高次多项式。"""
import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 100))
    y = np.sin(4 * x) + 0.15 * rng.standard_normal(100)
    xx = np.linspace(0, 1, 300)
    cs = CubicSpline(x, y, bc_type="natural")
    poly = LinearRegression().fit(PolynomialFeatures(30).fit_transform(x[:, None]), y)
    plt.scatter(x, y, s=8, alpha=0.4)
    plt.plot(xx, cs(xx), label="三次样条(natural)")
    plt.plot(xx, poly.predict(PolynomialFeatures(30).fit_transform(xx[:, None])), label="30次多项式(过拟合)")
    plt.legend()
    plt.title("分段样条 vs 全局多项式")
    plt.savefig("07_Basis_Expansions_and_Regularization/7.2_Piecewise_Polynomials_and_Splines/7.2_spline.png", dpi=110)
    print("ok: 7.2")


if __name__ == "__main__":
    main()