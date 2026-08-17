# 6.7_Exercises
"""习题: 比较 LDA 与逻辑回归在鸢尾花上的准确率。"""
import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    for name, cls in [("LDA", LinearDiscriminantAnalysis()),
                      ("逻辑回归", LogisticRegression(max_iter=1000))]:
        cls.fit(Xtr, ytr)
        print(f"{name} 测试准确率 = {cls.score(Xte, yte):.3f}")


if __name__ == "__main__":
    main()