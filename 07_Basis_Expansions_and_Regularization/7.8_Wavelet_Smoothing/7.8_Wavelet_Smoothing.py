# 7.8_Wavelet_Smoothing
"""小波平滑: 手写 Haar 小波硬阈值去噪。"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def haar_fwd(x):
    n = len(x)
    x = x.copy()
    c = []
    while n > 1:
        half = n // 2
        diffs = (x[:half] - x[half:2 * half]) / 2
        c.append(diffs)
        x = (x[:half] + x[half:2 * half]) / 2
        n = half
    return x, c


def haar_inv(mean, cs):
    x = np.array([mean])
    for d in reversed(cs):
        n = 2 * len(x)
        y = np.empty(n)
        y[0::2] = x + d
        y[1::2] = x - d
        x = y
    return x


def main() -> None:
    n = 128
    t = np.linspace(0, 1, n)
    clean = np.where(t < 0.5, 0.0, 1.0)
    noisy = clean + 0.3 * rng.standard_normal(n)
    mean, cs = haar_fwd(noisy)
    lam = 0.35
    cs_t = [np.where(np.abs(d) > lam, d, 0.0) for d in cs]
    denoised = haar_inv(mean, cs_t)
    mse = np.mean((denoised - clean) ** 2)
    plt.plot(t, noisy, alpha=0.5, label="带噪")
    plt.plot(t, denoised, lw=2, label="Haar 阈值去噪")
    plt.plot(t, clean, "k--", label="真实")
    plt.legend()
    plt.title(f"小波(去噪后 MSE={mse:.3f})")
    plt.savefig("07_Basis_Expansions_and_Regularization/7.8_Wavelet_Smoothing/7.8_wavelet.png", dpi=110)
    print(f"ok: 7.8 MSE={mse:.3f}")


if __name__ == "__main__":
    main()