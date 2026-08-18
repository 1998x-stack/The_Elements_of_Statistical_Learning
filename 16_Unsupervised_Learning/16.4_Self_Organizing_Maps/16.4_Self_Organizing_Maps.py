# 16.4_Self_Organizing_Maps
"""自组织映射(SOM)示意: 用原型/网格质心近似拓扑映射。"""
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    rng = np.random.default_rng(0)
    X, _ = make_blobs(n_samples=300, centers=5, cluster_std=1.0, random_state=0)
    # 用 k-means(网格 ~ 原型)近似 SOM 的原型映射
    K = 9
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
    print(f"SOM 示意: 用 k={K} 个原型(网格)表示数据; 保持拓扑(邻接原型较近)。")
    print("惯性:", round(km.inertia_, 2))


if __name__ == "__main__":
    main()