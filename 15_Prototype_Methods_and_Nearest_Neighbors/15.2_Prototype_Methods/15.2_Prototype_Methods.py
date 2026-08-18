# 15.2_Prototype_Methods
"""原型方法: 每类用均值原型, 按最近原型分类。"""
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.2, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    nc = NearestCentroid().fit(Xtr, ytr)
    print("原型(每类均值)分类 测试准确率 =", round(accuracy_score(yte, nc.predict(Xte)), 3))
    print("决策边界是原型间的垂直平分面(Voronoi)。")


if __name__ == "__main__":
    main()