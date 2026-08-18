# 15.5_Computational_Considerations
"""计算考虑: 暴力近邻 vs sklearn 最近邻查询耗时。"""
import time
import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore", category=RuntimeWarning)


def brute(query, X, k):
    d = np.linalg.norm(X[None, :, :] - query[:, None, :], axis=2)
    return np.argsort(d, axis=1)[:, :k]


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 10))
    q = rng.normal(size=(200, 10))
    k = 5
    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(X)
    nn.kneighbors(q)
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    brute(q, X, k)
    t3 = time.perf_counter()
    print(f"KD-tree 查询: {t1-t0:.4f}s   暴力: {t3-t2:.4f}s")
    print("KD树 / 球树 在大数据上比暴力快; 高维时树退化。")


if __name__ == "__main__":
    main()