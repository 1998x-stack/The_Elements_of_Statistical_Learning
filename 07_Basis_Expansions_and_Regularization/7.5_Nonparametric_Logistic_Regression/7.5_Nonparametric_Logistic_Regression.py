# 7.5_Nonparametric_Logistic_Regression
"""非参数逻辑回归: 用样条基构造 logit 再做逻辑回归。"""
import warnings
import numpy as np
from scipy.interpolate import BSpline
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def spline_basis(x, n_knots=6):
    knots = np.linspace(x.min(), x.max(), n_knots + 2)[1:-1]
    return BSpline.design_matrix(x, knots, k=3)


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 300))
    p = 1 / (1 + np.exp(-np.sin(4 * x) * 4))
    y = (rng.random(300) < p).astype(int)
    X = spline_basis(x).toarray()
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    xx = np.linspace(0, 1, 300)
    prob = clf.predict_proba(spline_basis(xx).toarray())[:, 1]
    plt.scatter(x, y, s=8, alpha=0.3)
    plt.plot(xx, prob, label="样条逻辑回归后验概率")
    plt.plot(xx, 1 / (1 + np.exp(-np.sin(10 * xx) * 4)), "--", label="真实")
    plt.legend()
    plt.title("非参数逻辑回归")
    plt.savefig("07_Basis_Expansions_and_Regularization/7.5_Nonparametric_Logistic_Regression/7.5_logistic.png", dpi=110)
    print("ok: 7.5")


if __name__ == "__main__":
    main()