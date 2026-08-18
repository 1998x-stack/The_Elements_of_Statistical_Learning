# 16.10_The_Google_PageRank_Algorithm
"""PageRank: 用幂迭代在小型邻接图上演算。"""
import numpy as np

# 邻接矩阵 (4 个网页)
A = np.array([[0, 1, 1, 0],
              [1, 0, 0, 0],
              [0, 1, 0, 1],
              [1, 0, 1, 0]], dtype=float)


def pagerank(adj, d=0.85, iters=100):
    n = adj.shape[0]
    out = adj.sum(axis=1)
    tele = (1 - d) / n
    P = np.full(n, 1.0 / n)
    for _ in range(iters):
        P = tele + d * np.array([sum(P[j] / out[j] for j in range(n) if adj[j, i] > 0)
                                 for i in range(n)])
    return P


def main() -> None:
    P = pagerank(A)
    print("PageRank 得分:", np.round(P, 3))
    print("高入度/重要的结点得分高; 由随机游走稳态定义。")


if __name__ == "__main__":
    main()