# 9.8_Minimum_Description_Length
"""最小描述长度: 用分段常数拟合比较"编码总长度"。"""
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def code_length(segments, y, yhat, n):
    """粗略总长度 = 参数数(分段) + 残差编码(正态对数似然比)."""
    d = len(segments)
    rss = np.sum((y - yhat) ** 2)
    return d * np.log2(n) + 0.5 * n * np.log2(rss + 1e-9)  # 简化表述


def main() -> None:
    n = 300
    x = np.sort(rng.uniform(0, 1, n))
    y = np.where(x < 0.5, 1.0, 2.0) + 0.2 * rng.standard_normal(n)
    print("分段数 描述长度(简化)")
    best = (None, np.inf)
    for k in (1, 2, 4, 8):
        bounds = np.linspace(0, 1, k + 1)
        idx = np.clip(np.digitize(x, bounds[1:-1]), 0, k - 1)
        yhat = np.array([y[idx == j].mean() for j in range(k)])[idx]
        L = code_length(bounds[1:-1], y, yhat, n)
        if L < best[1]:
            best = (k, L)
        print(f"   {k}     {L:.2f}")
    print("最优分段数:", best[0])


if __name__ == "__main__":
    main()