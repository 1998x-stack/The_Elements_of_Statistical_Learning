# 8.5_Local_Likelihood_and_Other_Models
"""局部似然: 核加权局部逻辑回归估计非线性后验。"""
import warnings
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def local_logistic(x, y, xx, h=0.3):
    out = np.empty_like(xx)
    for j, t in enumerate(xx):
        w = np.exp(-((x - t) / h) ** 2)
        def negll(beta):
            p = 1 / (1 + np.exp(-(beta[0] + beta[1] * (x - t))))
            return -np.sum(w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
        res = minimize(negll, [0.0, 0.0], method="Powell")
        out[j] = 1 / (1 + np.exp(-res.x[0]))
    return out


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 300))
    p = 1 / (1 + np.exp(-np.sin(5 * x) * 4))
    y = (rng.random(300) < p).astype(int)
    xx = np.linspace(0, 1, 200)
    plt.scatter(x, y, s=6, alpha=0.3)
    plt.plot(xx, local_logistic(x, y, xx, 0.2), label="局部逻辑回归")
    plt.plot(xx, 1 / (1 + np.exp(-np.sin(5 * xx) * 4)), "k--", label="真实")
    plt.legend()
    plt.title("局部似然: 局部逻辑回归")
    plt.savefig("08_Kernel_Smoothing_Methods/8.5_Local_Likelihood_and_Other_Models/8.5_locallik.png", dpi=110)
    print("ok: 8.5")


if __name__ == "__main__":
    main()