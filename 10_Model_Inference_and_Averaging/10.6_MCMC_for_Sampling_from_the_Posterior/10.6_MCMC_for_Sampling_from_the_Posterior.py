# 10.6_MCMC_for_Sampling_from_the_Posterior
"""MCMC: Metropolis-Hastings 采样简单后验。"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    # 目标后验 ~ N(2,1)(未归一), 用随机游走 MH 采样
    target_mean, target_sd = 2.0, 1.0
    n = 5000
    x = np.empty(n)
    x[0] = 0.0
    acc = 0
    for t in range(1, n):
        prop = x[t - 1] + rng.normal(0, 0.8)
        # 接受概率 = 后验比例
        r = np.exp(-0.5 * ((prop - target_mean) / target_sd) ** 2
                   + 0.5 * ((x[t - 1] - target_mean) / target_sd) ** 2)
        if rng.random() < min(1.0, r):
            x[t] = prop
            acc += 1
        else:
            x[t] = x[t - 1]
    burn = x[1000:]
    print(f"采样均值={burn.mean():.3f} (真=2)  接受率={acc/n:.2f}")
    plt.hist(burn, bins=50, density=True, alpha=0.6)
    plt.title("MH 采样出的后验分布")
    plt.savefig("10_Model_Inference_and_Averaging/10.6_MCMC_for_Sampling_from_the_Posterior/10.6_mcmc.png", dpi=110)
    print("ok: 10.6")


if __name__ == "__main__":
    main()