# 14.9_Exercises
"""习题: SVM 核与参数(C,gamma)对双月分类测试误差的影响。"""
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = make_moons(n_samples=300, noise=0.15, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    print("kernel  C/gamma  测试准确率")
    for kern, kwargs in [("linear", {}), ("rbf", {"gamma": 1}), ("rbf", {"gamma": 20}),
                         ("poly", {"degree": 3})]:
        score = SVC(kernel=kern, C=1.0, **kwargs).fit(Xtr, ytr).score(Xte, yte)
        print(f"  {kern:6s}  {str(list(kwargs.values())):8s}  {score:.3f}")


if __name__ == "__main__":
    main()