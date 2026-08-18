# 15.7_Exercises
"""习题: k 与距离度量对 2 维 kNN 的影响。"""
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_classification(n_samples=300, n_features=4, n_informative=3, random_state=0)
    print("k   CV准确率")
    for k in (1, 5, 20, 60):
        s = cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5).mean()
        print(f"  {k:3d}    {s:.3f}")
    print("k 小→灵活高方差; k 大→平滑高偏差。")


if __name__ == "__main__":
    main()