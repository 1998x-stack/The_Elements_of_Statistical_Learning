# 5.4_Shrinkage_Methods
"""收缩方法: 岭回归系数的路径与 Lasso 稀疏解对比。"""
import warnings
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    X, y = make_regression(n_samples=80, n_features=8, n_informative=5, noise=0.5, random_state=0)
    alphas = np.logspace(-3, 3, 40)
    ridge_coefs = [Ridge(alpha=a).fit(X, y).coef_ for a in alphas]
    plt.plot(np.log10(alphas), np.array(ridge_coefs))
    plt.xlabel("log10(alpha)")
    plt.ylabel("系数")
    plt.title("岭回归系数路径")
    plt.savefig("05_Linear_Methods_for_Regression/5.4_Shrinkage_Methods/5.4_岭路径.png", dpi=110)
    lasso = Lasso(alpha=0.5, max_iter=5000).fit(X, y)
    print("Lasso 非零系数个数:", int(np.sum(lasso.coef_ != 0)))
    print("ok: 5.4")


if __name__ == "__main__":
    main()