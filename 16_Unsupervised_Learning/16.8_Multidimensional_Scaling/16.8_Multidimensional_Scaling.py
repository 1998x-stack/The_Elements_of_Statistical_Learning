# 16.8_Multidimensional_Scaling
"""多维尺度(MDS): 从距离阵嵌入到低维。"""
import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X = load_iris().data
    D = pairwise_distances(X)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0).fit(D)
    print("MDS 嵌入形状:", mds.embedding_.shape)
    print("应力(stress):", round(mds.stress_, 2))
    print("MDS 保留样本间距离的结构。")


if __name__ == "__main__":
    main()