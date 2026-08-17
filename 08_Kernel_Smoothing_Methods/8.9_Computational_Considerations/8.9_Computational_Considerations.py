# 8.9_Computational_Considerations
"""计算考虑: 核平滑的复杂度对比(直接循环 vs 稀疏/向量化)。"""
import time
import numpy as np

rng = np.random.default_rng(0)


def main() -> None:
    N = 2000
    M = 500
    x = rng.uniform(0, 1, N)
    y = np.sin(6 * x) + 0.2 * rng.standard_normal(N)
    xx = rng.uniform(0, 1, M)
    h = 0.1
    t0 = time.perf_counter()
    out = np.empty(M)
    for j, t in enumerate(xx):
        w = np.exp(-((x - t) / h) ** 2)
        out[j] = np.sum(w * y) / np.sum(w)
    t1 = time.perf_counter()
    # 向量化: 对所有测试点同时计算权重矩阵 (内存大但快)
    D = (x[None, :] - xx[:, None]) / h
    W = np.exp(-D ** 2)
    out2 = (W @ y) / W.sum(axis=1)
    t2 = time.perf_counter()
    print(f"循环: {t1-t0:.4f}s   向量化(权重矩阵): {t2-t1:.4f}s")
    print("两者最大差异:", np.abs(out - out2).max())


if __name__ == "__main__":
    main()