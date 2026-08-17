# 8.2_Selecting_the_Width_of_the_Kernel
"""选择核宽: 交叉验证在带宽网格上选最优 h。"""
import warnings
import numpy as np
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def gauss_smooth(x, y, xx, h):
    out = np.empty_like(xx)
    for j, t in enumerate(xx):
        w = np.exp(-((x - t) / h) ** 2)
        out[j] = np.sum(w * y) / np.sum(w)
    return out


def main() -> None:
    x = np.sort(rng.uniform(0, 1, 150))
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(150)
    hs = np.linspace(0.03, 0.5, 40)
    kf = KFold(5, shuffle=True, random_state=0)
    cv_err = []
    for h in hs:
        err = 0.0
        for tr, va in kf.split(x):
            xt, yt = x[tr], y[tr]
            xv, yv = x[va], y[va]
            pred = gauss_smooth(xt, yt, xv, h)
            err += np.mean((yv - pred) ** 2)
        cv_err.append(err / 5)
    best = hs[np.argmin(cv_err)]
    plt.plot(hs, cv_err, "-o", ms=3)
    plt.axvline(best, color="r", ls="--", label=f"最优 h={best:.3f}")
    plt.xlabel("带宽 h"); plt.ylabel("5折 CV 误差")
    plt.legend(); plt.title("带宽选择 (交叉验证)")
    plt.savefig("08_Kernel_Smoothing_Methods/8.2_Selecting_the_Width_of_the_Kernel/8.2_CV带宽.png", dpi=110)
    print(f"最优带宽 h* = {best:.3f}")


if __name__ == "__main__":
    main()