# 10.5_The_EM_Algorithm
"""EM 算法: 手写二分量高斯混合, 显示对数似然单调递增。"""
import warnings
import numpy as np
from scipy.stats import norm
warnings.filterwarnings("ignore", category=RuntimeWarning)

rng = np.random.default_rng(0)


def estep(x, m1, m2, s1, s2, pi):
    p1 = pi * norm.pdf(x, m1, s1)
    p2 = (1 - pi) * norm.pdf(x, m2, s2)
    return p1 / (p1 + p2 + 1e-12)  # 责任(属类1的后验)


def loglik(x, m1, m2, s1, s2, pi):
    return np.sum(np.log(pi * norm.pdf(x, m1, s1) + (1 - pi) * norm.pdf(x, m2, s2) + 1e-12))


def main() -> None:
    n = 200
    x = np.concatenate([rng.normal(0, 1, n), rng.normal(4, 1.5, n)])
    m1, m2 = 1.0, 3.0
    s1, s2, pi = 1.0, 1.0, 0.5
    ll = loglik(x, m1, m2, s1, s2, pi)
    print("迭代  对数似然")
    for it in range(30):
        g = estep(x, m1, m2, s1, s2, pi)   # E步: 责任
        m1 = (g * x).sum() / g.sum()       # M步: 均值
        m2 = ((1 - g) * x).sum() / (1 - g).sum()
        s1 = np.sqrt((g * (x - m1) ** 2).sum() / g.sum())
        s2 = np.sqrt(((1 - g) * (x - m2) ** 2).sum() / (1 - g).sum())
        pi = g.mean()
        ll = loglik(x, m1, m2, s1, s2, pi)
        if it % 5 == 0 or it == 29:
            print(f"{it:3d}   {ll:.3f}")
    print("最终 log-likelihood =", round(ll, 4), "  两均值 =", np.round([m1, m2], 2))


if __name__ == "__main__":
    main()