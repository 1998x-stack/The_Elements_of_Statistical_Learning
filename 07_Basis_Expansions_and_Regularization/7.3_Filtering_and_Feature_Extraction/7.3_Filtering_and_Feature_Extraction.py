# 7.3_Filtering_and_Feature_Extraction
"""滤波与特征提取: 对信号列做滑动平均(卷积)平滑。"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def moving_average(signal, k=7):
    k = np.ones(k) / k
    return np.convolve(signal, k, mode="same")


def main() -> None:
    n = 200
    t = np.linspace(0, 1, n)
    signal = np.sin(6 * np.pi * t) + 0.4 * rng.standard_normal(n)
    plt.plot(t, signal, alpha=0.5, label="原信号(带噪)")
    plt.plot(t, moving_average(signal, 9), lw=2, label="滑动平均(卷积平滑)")
    plt.legend()
    plt.title("滤波/特征提取: 卷积平滑")
    plt.savefig("07_Basis_Expansions_and_Regularization/7.3_Filtering_and_Feature_Extraction/7.3_filtering.png", dpi=110)
    print("ok: 7.3")


if __name__ == "__main__":
    main()