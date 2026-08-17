# 11.3_PRIM:_Bump_Hunting
"""PRIM: 贪心寻找高响应区域的盒子。"""
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def main() -> None:
    n = 2000
    X = rng.uniform(0, 1, (n, 2))
    y = (np.abs(X[:, 0] - 0.5) < 0.2) & (np.abs(X[:, 1] - 0.5) < 0.2)
    y = y.astype(float) + 0.05 * rng.standard_normal(n)
    # 原始冒泡: 逐步剥去边缘(去掉均值最小的一边), 选响应最高的盒
    lo, hi = np.zeros(2), np.ones(2)
    best = None
    for _ in range(40):
        mask = np.all((X[:, 0] >= lo[0]) & (X[:, 0] <= hi[0]) &
                      (X[:, 1] >= lo[1]) & (X[:, 1] <= hi[1]), axis=0)
        if mask.sum() < 20:
            break
        mean = y[mask].mean()
        if best is None or mean > best[0]:
            best = (mean, (lo.copy(), hi.copy()), mask.sum())
        # 剥掉平均增量最小的一个边
        cand = []
        for d in range(2):
            c1 = lo.copy(); c1[d] = np.quantile(X[mask, d], 0.1); cand.append((y[mask & (X[:, d] >= c1[d])].mean(), (c1, hi.copy())))
            c2 = hi.copy(); c2[d] = np.quantile(X[mask, d], 0.9); cand.append((y[mask & (X[:, d] <= c2[d])].mean(), (lo.copy(), c2)))
        # 选取添加响应最大的剥边方式
        (_, (lo, hi)) = max(cand, key=lambda t: t[0])
    print("PRIM 找到高响应盒(×高=%.2f) 覆盖 %d 点" % (best[0], best[2]))
    print("盒子范围 x1∈[%.2f,%.2f] x2∈[%.2f,%.2f]" % (best[1][0][0], best[1][1][0], best[1][0][1], best[1][1][1]))


if __name__ == "__main__":
    main()