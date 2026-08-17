# 3.3_Statistical_Decision_Theory
"""统计决策理论: 分箱估计回归函数 E[Y|X]。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(0)


def main() -> None:
    n = 800
    x = rng.uniform(0, 4, n)
    y = np.sin(1.5 * x) + 0.15 * rng.standard_normal(n)
    xs = np.sort(x)
    ys = y[np.argsort(x)]
    nbins = 20
    bins = np.linspace(0, 4, nbins + 1)
    idx = np.digitize(xs, bins) - 1
    centers = (bins[:-1] + bins[1:]) / 2
    est = [ys[idx == k].mean() if np.any(idx == k) else np.nan for k in range(nbins)]
    xx = np.linspace(0, 4, 300)
    plt.scatter(xs, ys, s=4, alpha=0.4, label="样本")
    plt.plot(xx, np.sin(1.5 * xx), "k--", label="真实 E[Y|X]")
    plt.plot(centers, est, "-o", label="分箱估计")
    plt.legend()
    plt.title("回归函数 E[Y|X] 的估计")
    plt.savefig("03_Introduction/3.3_Statistical_Decision_Theory/3.3_回归函数.png", dpi=110)
    print("ok: 3.3")


if __name__ == "__main__":
    main()