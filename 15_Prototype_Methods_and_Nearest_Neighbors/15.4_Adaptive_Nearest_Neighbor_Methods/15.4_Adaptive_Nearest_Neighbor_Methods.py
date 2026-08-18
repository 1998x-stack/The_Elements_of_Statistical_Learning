# 15.4_Adaptive_Nearest_Neighbor_Methods
"""自适应最近邻: 用基于密集度/椭楯的距离度量示意。"""
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3, random_state=0)
    # 标准 Euclid kNN vs 标准化(特征缩放=一种自适应)
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    cv_e = cross_val_score(KNeighborsClassifier(15), X, y, cv=5).mean()
    cv_s = cross_val_score(KNeighborsClassifier(15), Xs, y, cv=5).mean()
    print(f"原始 Euclid kNN CV 准确率 = {cv_e:.3f}")
    print(f"标准化(自适应度量) kNN = {cv_s:.3f}")
    print("自适应方法(特征加权/椭球度量)可显著改善高维 kNN。")


if __name__ == "__main__":
    main()