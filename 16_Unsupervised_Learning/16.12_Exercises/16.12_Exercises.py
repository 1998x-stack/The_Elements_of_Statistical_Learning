# 16.12_Exercises
"""习题: k-means 的 K 选择(肘法)演示。"""
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=0)
    print("K  惯性(sse)")
    inert = []
    for K in range(1, 8):
        s = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X).inertia_
        inert.append((K, round(s, 1)))
        print(f"  {K}   {round(s,1)}")
    print("肘部位于真实类数 K=4 附近, 之后惯性下降放缓。")


if __name__ == "__main__":
    main()